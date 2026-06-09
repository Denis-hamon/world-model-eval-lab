"""Numerical equivalence: wmel's DreamerV3 adapter vs upstream dreamerv3-torch.

This is the test behind the adapter's "verified numerically equivalent"
claim: build the upstream RSSM/encoder/decoder at a small size, copy their
weights through `port_from_dreamerv3_torch`, and require the adapter's
`(obs, action) -> next_obs` to match the upstream path

    obs_step(is_first=True, sample=False) -> img_step(sample=False)
    -> decoder mode (symexp)

to float32 precision.

It needs the upstream source tree, so it skips itself unless
`third_party/dreamerv3-torch/` is present (run `./scripts/setup_dreamerv3.sh`)
and its import-time deps (numpy, tensorboard) are installed. CI does not run
it; the GPU box pre-flight does (see experiments/GPU_ROADMAP.md, Task 6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UPSTREAM = _REPO_ROOT / "third_party" / "dreamerv3-torch"

if not (_UPSTREAM / "networks.py").exists():
    pytest.skip(
        "third_party/dreamerv3-torch not present (run scripts/setup_dreamerv3.sh)",
        allow_module_level=True,
    )

sys.path.insert(0, str(_UPSTREAM))
networks = pytest.importorskip("networks")

from torch import nn  # noqa: E402

from wmel.adapters.dreamerv3_adapter import port_from_dreamerv3_torch  # noqa: E402

# Upstream's MLP defaults to device="cuda" just to build a std tensor that
# the symlog_mse decoder head never uses; force the default to cpu so the
# test runs on any box.
networks.MLP.__init__.__defaults__ = tuple(
    "cpu" if d == "cuda" else d for d in networks.MLP.__init__.__defaults__
)

OBS_DIM, ACT_DIM = 6, 1
STOCH, DISCRETE, DETER, HIDDEN = 4, 4, 16, 16
UNITS, LAYERS = 24, 2
SHAPES = {"orientations": (4,), "velocity": (2,)}

ARCH = {
    "obs_dim": OBS_DIM,
    "action_dim": ACT_DIM,
    "stoch": STOCH,
    "discrete": DISCRETE,
    "deter": DETER,
    "hidden": HIDDEN,
    "encoder_layers": LAYERS,
    "encoder_units": UNITS,
    "decoder_layers": LAYERS,
    "decoder_units": UNITS,
    "symlog_inputs": True,
}


class _UpstreamWorldModel(nn.Module):
    """The encoder/dynamics/decoder trio exactly as models.WorldModel builds
    them for a dmc_proprio config, at test size."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = networks.MultiEncoder(
            SHAPES, mlp_keys=".*", cnn_keys="$^", act="SiLU", norm=True,
            cnn_depth=32, kernel_size=4, minres=4, mlp_layers=LAYERS,
            mlp_units=UNITS, symlog_inputs=True,
        )
        self.dynamics = networks.RSSM(
            stoch=STOCH, deter=DETER, hidden=HIDDEN, rec_depth=1,
            discrete=DISCRETE, act="SiLU", norm=True, mean_act="none",
            std_act="sigmoid2", min_std=0.1, unimix_ratio=0.01,
            initial="learned", num_actions=ACT_DIM,
            embed=self.encoder.outdim, device="cpu",
        )
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(
            STOCH * DISCRETE + DETER, SHAPES, mlp_keys=".*", cnn_keys="$^",
            act="SiLU", norm=True, cnn_depth=32, kernel_size=4, minres=4,
            mlp_layers=LAYERS, mlp_units=UNITS, cnn_sigmoid=False,
            image_dist="mse", vector_dist="symlog_mse", outscale=1.0,
        )


def test_adapter_matches_upstream_forward_exactly() -> None:
    torch.manual_seed(7)
    wm = _UpstreamWorldModel().eval()
    # Upstream init leaves W at zero and several heads near zero; randomise
    # so the comparison exercises non-trivial values everywhere.
    with torch.no_grad():
        for p in wm.parameters():
            p.copy_(torch.randn_like(p) * 0.3)

    adapter = port_from_dreamerv3_torch(wm.state_dict(), ARCH).eval()

    obs = torch.randn(1, OBS_DIM)
    action = torch.tensor([[0.5]])

    with torch.no_grad():
        obs_dict = {"orientations": obs[:, :4], "velocity": obs[:, 4:]}
        embed = wm.encoder(obs_dict)
        is_first = torch.ones(1)
        post, _prior = wm.dynamics.obs_step(None, None, embed, is_first, sample=False)
        prior2 = wm.dynamics.img_step(post, action, sample=False)
        feat = wm.dynamics.get_feat(prior2)
        dists = wm.heads["decoder"](feat)
        upstream_next = torch.cat(
            [dists["orientations"].mode(), dists["velocity"].mode()], dim=-1
        )

        adapter_next = adapter(obs, action)

    assert torch.allclose(adapter_next, upstream_next, atol=1e-5), (
        f"adapter {adapter_next.tolist()} != upstream {upstream_next.tolist()}"
    )
