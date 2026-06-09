"""DreamerV3 dynamics adapter for the wmel planner contract.

This is the second published-world-model adapter shipped by the framework
(after TD-MPC2 in v0.12). The point is to plug DreamerV3's RSSM world model
into `TabularWorldModelPlanner` without dragging its training stack
(dreamerv3-torch, gym 0.22, tensorboard, ...) into `src/wmel/`.

Approach: re-implement the minimal RSSM building blocks needed to load
weights from the reference PyTorch implementation (NM512/dreamerv3-torch):
the proprioceptive MLP encoder with symlog inputs, the GRU-based recurrent
cell with layer norm, the prior/posterior sufficient-statistics heads over
32x32 discrete latents, and the symlog-MSE observation decoder that DreamerV3
ships natively (unlike TD-MPC2, no post-hoc decoder training is needed).
`port_from_dreamerv3_torch` maps an upstream world-model `state_dict` onto
this module; the experiment script (`experiments/dmc_acrobot/dreamerv3_cpg.py`)
knows how to *produce* that state dict by training upstream.

Contract and an honest limitation
---------------------------------
The factory returns a `(state, action) -> next_state` callable, the same
contract as `wmel.adapters.tdmpc2_adapter.make_tdmpc2_dynamics`. DreamerV3,
however, is *recurrent*: its latent state `(deter, stoch)` accumulates
history that a Markovian `(obs, action)` interface cannot carry. Each call
therefore performs a one-frame *Markovian projection*:

    1. embed the observation (first-frame posterior: the learned initial
       deterministic state plus one zero-action GRU step, exactly how
       dreamerv3-torch processes `is_first=True` frames),
    2. take one imagination step with the given action,
    3. decode the resulting `(deter, stoch)` features back to obs space.

This is the same compromise the TD-MPC2 adapter makes by design (TD-MPC2 is
Markovian in its latent), applied to a recurrent model: memory beyond one
step is truncated. The CPG protocol measures what that projection costs the
planner; reporting it as a documented limitation is part of the result, not
a bug. All stochastic nodes use the distribution mode, so the callable is
deterministic and pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wmel.adapters.dreamerv3_adapter requires PyTorch. "
        "Install with `pip install -e \".[learned]\"`."
    ) from exc


Observation = Tuple[float, ...]
Action = Tuple[float, ...]

# LayerNorm epsilon used everywhere by dreamerv3-torch (networks.py).
_LN_EPS = 1e-3


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|). DreamerV3's input/output squashing."""
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `symlog`: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _normed_silu_block(in_dim: int, out_dim: int) -> list[nn.Module]:
    """Linear(bias=False) + LayerNorm(eps=1e-3) + SiLU.

    Must match dreamerv3-torch's MLP / `_img_in_layers` / `_img_out_layers` /
    `_obs_out_layers` layout exactly so that ported weights stay aligned:
    each block contributes 3 consecutive entries to an `nn.Sequential`.
    """
    return [
        nn.Linear(in_dim, out_dim, bias=False),
        nn.LayerNorm(out_dim, eps=_LN_EPS),
        nn.SiLU(),
    ]


def _build_trunk(in_dim: int, units: int, layers: int) -> nn.Sequential:
    """`layers` normed-SiLU blocks; first maps `in_dim -> units`, the rest
    `units -> units`. Mirrors dreamerv3-torch's `networks.MLP.layers`."""
    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        blocks.extend(_normed_silu_block(dim, units))
        dim = units
    return nn.Sequential(*blocks)


class DreamerV3Dynamics(nn.Module):
    """DreamerV3 RSSM (encoder + recurrent latent dynamics + obs decoder),
    restricted to proprioceptive (vector) observations and the discrete
    32x32 latent configuration that DreamerV3 uses on every benchmark.

    The layer layouts mirror dreamerv3-torch's `networks.RSSM`,
    `networks.MultiEncoder._mlp`, and `networks.MultiDecoder._mlp` so that
    `port_from_dreamerv3_torch` can copy weights one-to-one. The per-key
    decoder mean heads of the upstream `ModuleDict` are fused into a single
    `nn.Linear` at port time (exact, because the heads are linear over a
    shared trunk).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stoch: int = 32,
        discrete: int = 32,
        deter: int = 512,
        hidden: int = 512,
        encoder_layers: int = 5,
        encoder_units: int = 1024,
        decoder_layers: int = 5,
        decoder_units: int = 1024,
        symlog_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch = stoch
        self.discrete = discrete
        self.deter = deter
        self.symlog_inputs = symlog_inputs

        stoch_flat = stoch * discrete
        feat_dim = stoch_flat + deter
        embed_dim = encoder_units

        self.encoder = _build_trunk(obs_dim, encoder_units, encoder_layers)
        self.img_in = nn.Sequential(*_normed_silu_block(stoch_flat + action_dim, hidden))
        self.gru_linear = nn.Linear(hidden + deter, 3 * deter, bias=False)
        self.gru_norm = nn.LayerNorm(3 * deter, eps=_LN_EPS)
        self.img_out = nn.Sequential(*_normed_silu_block(deter, hidden))
        self.obs_out = nn.Sequential(*_normed_silu_block(deter + embed_dim, hidden))
        self.imgs_stat = nn.Linear(hidden, stoch_flat)
        self.obs_stat = nn.Linear(hidden, stoch_flat)
        # DreamerV3's learned initial deterministic state (`initial: learned`):
        # deter_0 = tanh(W).
        self.w_initial = nn.Parameter(torch.zeros(1, deter))
        self.decoder = _build_trunk(feat_dim, decoder_units, decoder_layers)
        self.decoder_mean = nn.Linear(decoder_units, obs_dim)

    def _onehot_mode(self, logits: torch.Tensor) -> torch.Tensor:
        """Mode of the 32x32 categorical latent, flattened.

        DreamerV3 mixes 1% uniform into the categorical (`unimix_ratio`);
        uniform mixing does not change the argmax, so the mode is the plain
        one-hot of the largest logit per group.
        """
        logit = logits.view(*logits.shape[:-1], self.stoch, self.discrete)
        idx = logit.argmax(dim=-1)
        one = F.one_hot(idx, num_classes=self.discrete).to(logits.dtype)
        return one.reshape(logits.shape)

    def _gru(self, x: torch.Tensor, deter: torch.Tensor) -> torch.Tensor:
        """dreamerv3-torch `GRUCell` with norm=True and update_bias=-1."""
        parts = self.gru_norm(self.gru_linear(torch.cat([x, deter], dim=-1)))
        reset, cand, update = torch.split(parts, self.deter, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Learned initial `(deter, stoch)`, as in `RSSM.initial("learned")`:
        deter_0 = tanh(W), stoch_0 = mode(prior_head(deter_0))."""
        deter = torch.tanh(self.w_initial).expand(batch_size, -1)
        stoch = self._onehot_mode(self.imgs_stat(self.img_out(deter)))
        return deter, stoch

    def _img_step(
        self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One imagination step: `RSSM.img_step` with `sample=False`."""
        x = self.img_in(torch.cat([stoch, action], dim=-1))
        deter_next = self._gru(x, deter)
        stoch_next = self._onehot_mode(self.imgs_stat(self.img_out(deter_next)))
        return deter_next, stoch_next

    def posterior_from_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """First-frame posterior `(deter, stoch)` for a raw observation.

        Replays how dreamerv3-torch handles `is_first=True`: start from the
        learned initial state, take one zero-action imagination step to get
        the prior deterministic state, then condition the posterior head on
        `(deter, embed(obs))`.
        """
        batch = obs.shape[0]
        x = symlog(obs) if self.symlog_inputs else obs
        embed = self.encoder(x)
        deter0, stoch0 = self.initial_state(batch)
        zero_action = torch.zeros(batch, self.action_dim, dtype=obs.dtype, device=obs.device)
        deter1, _ = self._img_step(deter0, stoch0, zero_action)
        post_logits = self.obs_stat(self.obs_out(torch.cat([deter1, embed], dim=-1)))
        return deter1, self._onehot_mode(post_logits)

    def decode(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Features -> observation. `symlog_mse` head: the prediction (mode)
        is `symexp` of the decoded mean."""
        feat = torch.cat([stoch, deter], dim=-1)
        return symexp(self.decoder_mean(self.decoder(feat)))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Markovian projection `(obs, action) -> next_obs`. See module
        docstring for what the projection keeps and what it truncates."""
        deter, stoch = self.posterior_from_obs(obs)
        deter_next, stoch_next = self._img_step(deter, stoch, action)
        return self.decode(deter_next, stoch_next)


def _strip_wm_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept either a bare WorldModel state_dict or a full Dreamer agent
    state_dict (keys prefixed `_wm.`); return the WorldModel view."""
    if any(k.startswith("_wm.") for k in state):
        return {k[len("_wm."):]: v for k, v in state.items() if k.startswith("_wm.")}
    return dict(state)


def discover_decoder_keys(state: Mapping[str, torch.Tensor]) -> list[str]:
    """Ordered observation keys of the upstream decoder's per-key mean heads.

    dreamerv3-torch's `MultiDecoder._mlp.mean_layer` is an `nn.ModuleDict`
    with one `nn.Linear` per observation key; `state_dict` preserves its
    insertion order, which is the order the encoder concatenates inputs in.
    """
    wm = _strip_wm_prefix(state)
    prefix = "heads.decoder._mlp.mean_layer."
    keys: list[str] = []
    for k in wm:
        if k.startswith(prefix) and k.endswith(".weight"):
            keys.append(k[len(prefix):-len(".weight")])
    if not keys:
        raise RuntimeError(
            "no decoder mean heads found under 'heads.decoder._mlp.mean_layer.'; "
            "is this a dreamerv3-torch world-model state_dict?"
        )
    return keys


def port_from_dreamerv3_torch(
    state: Mapping[str, torch.Tensor],
    arch: dict,
    decoder_keys: Sequence[str] | None = None,
) -> DreamerV3Dynamics:
    """Map a dreamerv3-torch world-model `state_dict` onto `DreamerV3Dynamics`.

    Parameters
    ----------
    state
        Either `agent.state_dict()` from a dreamerv3-torch `Dreamer` (keys
        prefixed `_wm.`) or `agent._wm.state_dict()`.
    arch
        Keyword arguments for `DreamerV3Dynamics.__init__`. Must match the
        upstream config the checkpoint was trained with (`dyn_stoch`,
        `dyn_deter`, `dyn_hidden`, `dyn_discrete`, encoder/decoder
        `mlp_layers` / `mlp_units`).
    decoder_keys
        Ordered observation keys to fuse the per-key decoder mean heads in.
        Defaults to `discover_decoder_keys(state)`. The fused order must be
        the order the wmel env flattens observations in (sorted keys for the
        `wmel.envs.dmc_*` environments); callers should assert that.

    Raises
    ------
    RuntimeError
        On any missing upstream key or shape mismatch. Loading is strict:
        a silent partial port would invalidate every CPG number downstream.
    """
    wm = _strip_wm_prefix(state)
    model = DreamerV3Dynamics(**arch)

    mapping: dict[str, str] = {}  # adapter key -> upstream key

    enc_layers = len(model.encoder) // 3
    for i in range(enc_layers):
        mapping[f"encoder.{3 * i}.weight"] = f"encoder._mlp.layers.Encoder_linear{i}.weight"
        mapping[f"encoder.{3 * i + 1}.weight"] = f"encoder._mlp.layers.Encoder_norm{i}.weight"
        mapping[f"encoder.{3 * i + 1}.bias"] = f"encoder._mlp.layers.Encoder_norm{i}.bias"

    mapping["img_in.0.weight"] = "dynamics._img_in_layers.0.weight"
    mapping["img_in.1.weight"] = "dynamics._img_in_layers.1.weight"
    mapping["img_in.1.bias"] = "dynamics._img_in_layers.1.bias"
    mapping["gru_linear.weight"] = "dynamics._cell.layers.GRU_linear.weight"
    mapping["gru_norm.weight"] = "dynamics._cell.layers.GRU_norm.weight"
    mapping["gru_norm.bias"] = "dynamics._cell.layers.GRU_norm.bias"
    mapping["img_out.0.weight"] = "dynamics._img_out_layers.0.weight"
    mapping["img_out.1.weight"] = "dynamics._img_out_layers.1.weight"
    mapping["img_out.1.bias"] = "dynamics._img_out_layers.1.bias"
    mapping["obs_out.0.weight"] = "dynamics._obs_out_layers.0.weight"
    mapping["obs_out.1.weight"] = "dynamics._obs_out_layers.1.weight"
    mapping["obs_out.1.bias"] = "dynamics._obs_out_layers.1.bias"
    mapping["imgs_stat.weight"] = "dynamics._imgs_stat_layer.weight"
    mapping["imgs_stat.bias"] = "dynamics._imgs_stat_layer.bias"
    mapping["obs_stat.weight"] = "dynamics._obs_stat_layer.weight"
    mapping["obs_stat.bias"] = "dynamics._obs_stat_layer.bias"
    mapping["w_initial"] = "dynamics.W"

    dec_layers = len(model.decoder) // 3
    for i in range(dec_layers):
        mapping[f"decoder.{3 * i}.weight"] = f"heads.decoder._mlp.layers.Decoder_linear{i}.weight"
        mapping[f"decoder.{3 * i + 1}.weight"] = f"heads.decoder._mlp.layers.Decoder_norm{i}.weight"
        mapping[f"decoder.{3 * i + 1}.bias"] = f"heads.decoder._mlp.layers.Decoder_norm{i}.bias"

    target = model.state_dict()
    ported: dict[str, torch.Tensor] = {}
    problems: list[str] = []
    for dst, src in mapping.items():
        if src not in wm:
            problems.append(f"missing upstream key {src!r} (for {dst!r})")
            continue
        if wm[src].shape != target[dst].shape:
            problems.append(
                f"shape mismatch for {dst!r}: upstream {src!r} has "
                f"{tuple(wm[src].shape)}, adapter expects {tuple(target[dst].shape)}"
            )
            continue
        ported[dst] = wm[src]

    # Fuse the per-key decoder mean heads into the single decoder_mean Linear.
    keys = list(decoder_keys) if decoder_keys is not None else discover_decoder_keys(wm)
    head_prefix = "heads.decoder._mlp.mean_layer."
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for key in keys:
        w_key, b_key = f"{head_prefix}{key}.weight", f"{head_prefix}{key}.bias"
        if w_key not in wm or b_key not in wm:
            problems.append(f"missing decoder mean head for obs key {key!r}")
            continue
        weights.append(wm[w_key])
        biases.append(wm[b_key])
    if not problems:
        fused_w = torch.cat(weights, dim=0)
        fused_b = torch.cat(biases, dim=0)
        if fused_w.shape != target["decoder_mean.weight"].shape:
            problems.append(
                f"fused decoder heads {keys} produce out_dim {fused_w.shape[0]}, "
                f"adapter expects obs_dim {model.obs_dim}"
            )
        else:
            ported["decoder_mean.weight"] = fused_w
            ported["decoder_mean.bias"] = fused_b

    if problems:
        raise RuntimeError(
            "cannot port dreamerv3-torch state_dict:\n  " + "\n  ".join(problems)
        )

    model.load_state_dict(ported, strict=True)
    return model


CheckpointDict = dict  # keys: "model_state", "arch", "action_levels" | "action_set", "meta"


def make_dreamerv3_dynamics(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Callable[[Observation, Action], Observation]:
    """Load a ported DreamerV3 checkpoint and return a `(state, action) ->
    next_state` callable compatible with
    `wmel.adapters.tabular_world_model.TabularWorldModelPlanner`.

    The checkpoint must have been produced by
    `experiments.dmc_acrobot.dreamerv3_cpg` (or any pipeline that follows the
    same dict schema, identical to the TD-MPC2 adapter's):

      - `model_state`: state_dict of `DreamerV3Dynamics`
      - `arch`: dict of `DreamerV3Dynamics.__init__` kwargs
      - one of:
        - `action_levels`: ordered tuple of floats for a 1-D action space.
          Each wmel action is the 1-tuple `(level,)`.
        - `action_set`: explicit list of action tuples for a multi-D action
          space. Used verbatim.

    The returned callable is pure and deterministic: every stochastic node
    uses the distribution mode, fresh tensors are allocated per call, and
    the model state is never mutated. Inference defaults to CPU for the same
    reason as the TD-MPC2 adapter: the planner makes thousands of
    single-state calls and GPU launch overhead would dominate.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    if "action_set" in ckpt:
        action_set = {tuple(float(x) for x in a) for a in ckpt["action_set"]}
    else:
        levels: Tuple[float, ...] = tuple(float(t) for t in ckpt["action_levels"])
        action_set = {(t,) for t in levels}

    model = DreamerV3Dynamics(**arch).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    @torch.no_grad()
    def _dynamics(state: Observation, action: Action) -> Observation:
        if action not in action_set:
            raise KeyError(
                f"action {action!r} is not in the checkpoint's discrete action set {sorted(action_set)}"
            )
        obs_t = torch.tensor([state], dtype=torch.float32, device=device)
        a_t = torch.tensor([list(action)], dtype=torch.float32, device=device)
        next_t = model(obs_t, a_t).squeeze(0)
        return tuple(float(x) for x in next_t)

    return _dynamics
