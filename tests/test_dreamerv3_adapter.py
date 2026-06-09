"""Tests for the DreamerV3 dynamics adapter.

Skipped automatically when torch is not installed (see `pytest.importorskip`),
so the rest of the suite keeps running on a stdlib-only checkout.

The porting tests build a *synthetic* dreamerv3-torch world-model state_dict
(correct key names and shapes, random values) rather than depending on the
upstream package: what the adapter must get right is the weight layout and
the deterministic forward contract, both of which are checkable without
training anything.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wmel.adapters.dreamerv3_adapter import (
    DreamerV3Dynamics,
    discover_decoder_keys,
    make_dreamerv3_dynamics,
    port_from_dreamerv3_torch,
    symexp,
    symlog,
)


# Tiny architecture: fast to build, same wiring as the real dmc_proprio one.
ARCH = {
    "obs_dim": 6,
    "action_dim": 1,
    "stoch": 4,
    "discrete": 4,
    "deter": 16,
    "hidden": 16,
    "encoder_layers": 2,
    "encoder_units": 24,
    "decoder_layers": 2,
    "decoder_units": 24,
    "symlog_inputs": True,
}

# DMC Acrobot proprio split: orientations (4) + velocity (2), sorted order,
# matching how `wmel.envs.dmc_acrobot` flattens observations.
DECODER_SHAPES = (("orientations", 4), ("velocity", 2))


def _synthetic_upstream_state(arch: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """A state_dict with dreamerv3-torch's exact key names and shapes."""
    g = torch.Generator().manual_seed(seed)

    def t(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * 0.1

    stoch_flat = arch["stoch"] * arch["discrete"]
    deter, hidden = arch["deter"], arch["hidden"]
    eu, du = arch["encoder_units"], arch["decoder_units"]
    feat = stoch_flat + deter

    state: dict[str, torch.Tensor] = {}
    in_dim = arch["obs_dim"]
    for i in range(arch["encoder_layers"]):
        state[f"encoder._mlp.layers.Encoder_linear{i}.weight"] = t(eu, in_dim)
        state[f"encoder._mlp.layers.Encoder_norm{i}.weight"] = t(eu)
        state[f"encoder._mlp.layers.Encoder_norm{i}.bias"] = t(eu)
        in_dim = eu

    state["dynamics._img_in_layers.0.weight"] = t(hidden, stoch_flat + arch["action_dim"])
    state["dynamics._img_in_layers.1.weight"] = t(hidden)
    state["dynamics._img_in_layers.1.bias"] = t(hidden)
    state["dynamics._cell.layers.GRU_linear.weight"] = t(3 * deter, hidden + deter)
    state["dynamics._cell.layers.GRU_norm.weight"] = t(3 * deter)
    state["dynamics._cell.layers.GRU_norm.bias"] = t(3 * deter)
    state["dynamics._img_out_layers.0.weight"] = t(hidden, deter)
    state["dynamics._img_out_layers.1.weight"] = t(hidden)
    state["dynamics._img_out_layers.1.bias"] = t(hidden)
    state["dynamics._obs_out_layers.0.weight"] = t(hidden, deter + eu)
    state["dynamics._obs_out_layers.1.weight"] = t(hidden)
    state["dynamics._obs_out_layers.1.bias"] = t(hidden)
    state["dynamics._imgs_stat_layer.weight"] = t(stoch_flat, hidden)
    state["dynamics._imgs_stat_layer.bias"] = t(stoch_flat)
    state["dynamics._obs_stat_layer.weight"] = t(stoch_flat, hidden)
    state["dynamics._obs_stat_layer.bias"] = t(stoch_flat)
    state["dynamics.W"] = t(1, deter)

    in_dim = feat
    for i in range(arch["decoder_layers"]):
        state[f"heads.decoder._mlp.layers.Decoder_linear{i}.weight"] = t(du, in_dim)
        state[f"heads.decoder._mlp.layers.Decoder_norm{i}.weight"] = t(du)
        state[f"heads.decoder._mlp.layers.Decoder_norm{i}.bias"] = t(du)
        in_dim = du

    for key, dim in DECODER_SHAPES:
        state[f"heads.decoder._mlp.mean_layer.{key}.weight"] = t(dim, du)
        state[f"heads.decoder._mlp.mean_layer.{key}.bias"] = t(dim)
    return state


def test_symlog_symexp_roundtrip() -> None:
    x = torch.tensor([-100.0, -1.0, -1e-4, 0.0, 1e-4, 1.0, 100.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_discover_decoder_keys_preserves_insertion_order() -> None:
    state = _synthetic_upstream_state(ARCH)
    assert discover_decoder_keys(state) == ["orientations", "velocity"]


def test_port_accepts_bare_and_agent_prefixed_state_dicts() -> None:
    state = _synthetic_upstream_state(ARCH)
    prefixed = {f"_wm.{k}": v for k, v in state.items()}
    # An agent state_dict also carries non-world-model entries; they must be ignored.
    prefixed["_task_behavior.actor.weight"] = torch.zeros(2, 2)

    bare = port_from_dreamerv3_torch(state, ARCH)
    full = port_from_dreamerv3_torch(prefixed, ARCH)
    for (k_a, v_a), (k_b, v_b) in zip(bare.state_dict().items(), full.state_dict().items()):
        assert k_a == k_b
        assert torch.equal(v_a, v_b)


def test_port_fuses_decoder_heads_in_key_order() -> None:
    state = _synthetic_upstream_state(ARCH)
    model = port_from_dreamerv3_torch(state, ARCH)
    expected_w = torch.cat(
        [state[f"heads.decoder._mlp.mean_layer.{k}.weight"] for k, _ in DECODER_SHAPES],
        dim=0,
    )
    expected_b = torch.cat(
        [state[f"heads.decoder._mlp.mean_layer.{k}.bias"] for k, _ in DECODER_SHAPES],
        dim=0,
    )
    assert torch.equal(model.decoder_mean.weight, expected_w)
    assert torch.equal(model.decoder_mean.bias, expected_b)


def test_port_rejects_missing_key() -> None:
    state = _synthetic_upstream_state(ARCH)
    del state["dynamics._cell.layers.GRU_linear.weight"]
    with pytest.raises(RuntimeError, match="GRU_linear"):
        port_from_dreamerv3_torch(state, ARCH)


def test_port_rejects_shape_mismatch() -> None:
    state = _synthetic_upstream_state(ARCH)
    state["dynamics.W"] = torch.zeros(1, ARCH["deter"] + 1)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        port_from_dreamerv3_torch(state, ARCH)


def test_port_rejects_wrong_decoder_fusion_dim() -> None:
    state = _synthetic_upstream_state(ARCH)
    with pytest.raises(RuntimeError, match="fused decoder heads"):
        port_from_dreamerv3_torch(state, ARCH, decoder_keys=["orientations"])


def _checkpoint(tmp_path, seed: int = 0):
    state = _synthetic_upstream_state(ARCH, seed=seed)
    model = port_from_dreamerv3_torch(state, ARCH)
    path = tmp_path / "dreamerv3_test.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": ARCH,
            "action_levels": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "meta": {"synthetic": True},
        },
        path,
    )
    return path


def test_factory_contract_types_and_determinism(tmp_path) -> None:
    dyn = make_dreamerv3_dynamics(_checkpoint(tmp_path))
    obs = (0.1, -0.2, 0.9, 0.8, 0.0, 0.3)
    out1 = dyn(obs, (0.5,))
    out2 = dyn(obs, (0.5,))

    assert isinstance(out1, tuple)
    assert len(out1) == ARCH["obs_dim"]
    assert all(isinstance(x, float) for x in out1)
    assert out1 == out2  # deterministic: every stochastic node uses the mode

    # A second factory from the same checkpoint must agree exactly.
    dyn_b = make_dreamerv3_dynamics(_checkpoint(tmp_path))
    assert dyn_b(obs, (0.5,)) == out1


def test_factory_rejects_unknown_action(tmp_path) -> None:
    dyn = make_dreamerv3_dynamics(_checkpoint(tmp_path))
    with pytest.raises(KeyError):
        dyn((0.0,) * 6, (0.25,))


def test_factory_supports_explicit_action_set(tmp_path) -> None:
    state = _synthetic_upstream_state({**ARCH, "action_dim": 2})
    model = port_from_dreamerv3_torch(state, {**ARCH, "action_dim": 2})
    path = tmp_path / "dreamerv3_2d.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": {**ARCH, "action_dim": 2},
            "action_set": [(-1.0, 0.0), (0.0, 1.0)],
            "meta": {"synthetic": True},
        },
        path,
    )
    dyn = make_dreamerv3_dynamics(path)
    out = dyn((0.0,) * 6, (-1.0, 0.0))
    assert len(out) == ARCH["obs_dim"]
    with pytest.raises(KeyError):
        dyn((0.0,) * 6, (1.0, 1.0))


def test_different_actions_change_the_prediction(tmp_path) -> None:
    """The action must actually flow through the imagination step."""
    dyn = make_dreamerv3_dynamics(_checkpoint(tmp_path))
    obs = (0.1, -0.2, 0.9, 0.8, 0.0, 0.3)
    assert dyn(obs, (-1.0,)) != dyn(obs, (1.0,))


def test_forward_batches_consistently() -> None:
    """Batched forward must equal per-sample forward (no cross-batch state)."""
    state = _synthetic_upstream_state(ARCH)
    model = port_from_dreamerv3_torch(state, ARCH).eval()
    obs = torch.randn(3, ARCH["obs_dim"])
    act = torch.randn(3, ARCH["action_dim"])
    with torch.no_grad():
        batched = model(obs, act)
        single = torch.stack([model(obs[i : i + 1], act[i : i + 1]).squeeze(0) for i in range(3)])
    assert torch.allclose(batched, single, atol=1e-6)
