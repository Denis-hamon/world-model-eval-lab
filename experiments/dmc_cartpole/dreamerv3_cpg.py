"""Train DreamerV3 on DMC Cartpole-swingup, then run the CPG protocol.

Task 7 (cross-env) counterpart of `experiments.dmc_acrobot.dreamerv3_cpg`:
the second published-model arm on a second environment, so that
`results/MODEL_TABLE.md` becomes a genuine model x env cross-comparison.
The CPG arm structure is preserved verbatim from
`experiments.dmc_cartpole.tdmpc2_cpg`: same `BenchmarkRunner`, same
`TabularWorldModelPlanner`, same `cartpole_upright_score`, same
`make_cartpole_oracle_dynamics`. Only the learned dynamics callable changes.

This mirrors the Acrobot DreamerV3 experiment one-for-one; see that module's
docstring for the full pipeline / setup / usage notes. Env-specific deltas
(per experiments/GPU_ROADMAP.md, Task 7):

    --task dmc_cartpole_swingup
    ARCH["obs_dim"]   = 5   (position 3 + velocity 2, sorted keys)
    ARCH["action_dim"] = 1
    5-level action grid, score/env from wmel.envs.dmc_cartpole
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Headless MuJoCo: must be set before any dm_control import.
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DREAMER_PKG = _REPO_ROOT / "third_party" / "dreamerv3-torch"
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    s = str(_entry)
    if _entry.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

import torch

from wmel.adapters.dreamerv3_adapter import (
    discover_decoder_keys,
    make_dreamerv3_dynamics,
    port_from_dreamerv3_torch,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_cartpole import (
    DMCCartpoleEnv,
    cartpole_upright_score,
    make_cartpole_oracle_dynamics,
)
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report

from experiments._seeding import eval_varied_factory


_RESULTS_DIR = _REPO_ROOT / "results" / "dmc_cartpole"


def default_workdir(seed: int) -> Path:
    """Per-seed training workdir. Upstream auto-resumes from
    `<workdir>/latest.pt`, so seeds must never share one."""
    return _RESULTS_DIR / f"dreamerv3_workdir_seed{seed}"


# dreamerv3-torch `dmc_proprio` architecture (configs.yaml defaults +
# dmc_proprio overrides). Must match the config the subprocess trains with;
# the port fails loudly on any mismatch.
ARCH = {
    "obs_dim": 5,        # cartpole: position (3) + velocity (2), sorted keys
    "action_dim": 1,
    "stoch": 32,
    "discrete": 32,
    "deter": 512,
    "hidden": 512,
    "encoder_layers": 5,
    "encoder_units": 1024,
    "decoder_layers": 5,
    "decoder_units": 1024,
    "symlog_inputs": True,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Smaller config for end-to-end validation.")
    p.add_argument("--steps", type=int, default=None, help="Override training steps (default: 500_000, smoke: 3_000).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0", help="Device for the training subprocess (CPG arms run on CPU).")
    p.add_argument(
        "--agent-checkpoint",
        type=Path,
        default=None,
        help="Port an existing dreamerv3-torch latest.pt instead of the default workdir one.",
    )
    p.add_argument("--skip-train", action="store_true", help="Skip training; port + CPG only.")
    p.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Training workdir override (default: results/dmc_cartpole/dreamerv3_workdir_seed<seed>).",
    )
    p.add_argument(
        "--out-suffix",
        default="",
        help="Suffix for the output JSON and adapter checkpoint, e.g. _seed1.",
    )
    p.add_argument(
        "--varied-init",
        action="store_true",
        help=(
            "Vary the initial state per episode (seed shared across arms) so "
            "success rates sample the task distribution. Off by default to "
            "reproduce fixed-init results."
        ),
    )
    return p.parse_args()


def _config(smoke: bool, steps_override: int | None) -> dict:
    if smoke:
        return {
            "training_steps": steps_override if steps_override is not None else 3_000,
            "num_candidates": 15,
            "plan_horizon": 8,
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
        }
    return {
        "training_steps": steps_override if steps_override is not None else 500_000,
        "num_candidates": 50,
        "plan_horizon": 15,
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def _train_dreamerv3(
    steps: int, seed: int, device: str, workdir: Path, smoke: bool
) -> None:
    """Run upstream training as a subprocess. Resumes from workdir/latest.pt."""
    dreamer_py = _DREAMER_PKG / "dreamer.py"
    if not dreamer_py.exists():
        raise FileNotFoundError(
            f"{dreamer_py} not found. Run ./scripts/setup_dreamerv3.sh first."
        )
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
    cmd = [
        sys.executable,
        str(dreamer_py),
        "--configs", "dmc_proprio",
        "--task", "dmc_cartpole_swingup",
        "--logdir", str(workdir),
        "--steps", str(steps),
        "--action_repeat", "1",      # like-for-like with wmel's 1-step oracle/env
        "--envs", "1",
        "--seed", str(seed),
        "--device", device,
        "--compile", "False",        # same fragility call as the TD-MPC2 script
        "--eval_episode_num", "0",   # we evaluate via wmel CPG, not upstream eval
        "--video_pred_log", "False",
    ]
    if smoke:
        cmd += ["--prefill", "500", "--eval_every", "1000", "--log_every", "500"]
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[train] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(_DREAMER_PKG), check=True)


def _port_agent_checkpoint(
    agent_ckpt: Path, checkpoint_path: Path, training_steps: int, seed: int
) -> None:
    """latest.pt -> adapter-format checkpoint at checkpoint_path."""
    print(f"[port] loading {agent_ckpt}")
    ckpt = torch.load(agent_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["agent_state_dict"] if "agent_state_dict" in ckpt else ckpt

    decoder_keys = discover_decoder_keys(state)
    # wmel's DMC envs flatten observation dicts in sorted-key order; the
    # ported encoder input / fused decoder output must use the same order.
    if decoder_keys != sorted(decoder_keys):
        raise RuntimeError(
            f"upstream obs keys {decoder_keys} are not in sorted order; "
            "the ported model would disagree with wmel's observation layout"
        )
    model = port_from_dreamerv3_torch(state, ARCH, decoder_keys=decoder_keys)

    action_levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "arch": ARCH,
            "action_levels": list(action_levels),
            "meta": {
                "training_steps": training_steps,
                "seed": seed,
                "frame_skip": 1,
                "source": "dreamerv3-torch dmc_proprio",
                "obs_keys": decoder_keys,
            },
        },
        checkpoint_path,
    )
    print(f"[ckpt] wrote {checkpoint_path.relative_to(_REPO_ROOT)}")


def main() -> None:
    args = _parse_args()
    cfg_dict = _config(smoke=args.smoke, steps_override=args.steps)
    seed = args.seed
    workdir = args.workdir if args.workdir is not None else default_workdir(seed)
    checkpoint_path = _RESULTS_DIR / f"dreamerv3_cartpole{args.out_suffix}.pt"
    json_path = _RESULTS_DIR / f"dreamerv3_cpg{args.out_suffix}.json"
    print(
        f"[setup] smoke={args.smoke}, training_steps={cfg_dict['training_steps']}, "
        f"workdir={workdir.name}, out_suffix={args.out_suffix!r}"
    )

    agent_ckpt = args.agent_checkpoint if args.agent_checkpoint is not None else workdir / "latest.pt"

    print("[1/4] Training DreamerV3 (upstream subprocess)...")
    if args.skip_train:
        print("  --skip-train: using existing checkpoint")
    else:
        _train_dreamerv3(
            cfg_dict["training_steps"],
            seed=seed,
            device=args.device,
            workdir=workdir,
            smoke=args.smoke,
        )

    print("[2/4] Porting world-model weights into the wmel adapter...")
    _port_agent_checkpoint(agent_ckpt, checkpoint_path, cfg_dict["training_steps"], seed)

    print("[3/4] Building wmel arms (oracle + DreamerV3 dynamics) and running benchmarks...")
    action_levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    env_template = DMCCartpoleEnv(discrete_levels=action_levels)

    def make_planner(dyn):
        return TabularWorldModelPlanner(
            dynamics=dyn,
            action_space=env_template.action_space,
            num_candidates=cfg_dict["num_candidates"],
            plan_horizon=cfg_dict["plan_horizon"],
            score=cartpole_upright_score,
            seed=seed,
        )

    # Both CPG arms must see identical per-episode initial states to stay
    # paired, so each arm builds its own factory from the SAME base seed.
    def make_eval_factory():
        if args.varied_init:
            return eval_varied_factory(DMCCartpoleEnv, seed, discrete_levels=action_levels)
        return lambda: DMCCartpoleEnv(discrete_levels=action_levels)

    oracle_planner = make_planner(make_cartpole_oracle_dynamics())
    oracle_results = BenchmarkRunner(
        env_factory=make_eval_factory(),
        policy=oracle_planner,
        episodes=cfg_dict["benchmark_episodes"],
        horizon=cfg_dict["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    oracle_card = compute_scorecard(
        oracle_results,
        policy_name="tabular-world-model (oracle dynamics)",
        compute_per_plan_call=oracle_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(oracle_card)

    dreamer_dyn = make_dreamerv3_dynamics(checkpoint_path, device="cpu")
    dreamer_planner = make_planner(dreamer_dyn)
    dreamer_results = BenchmarkRunner(
        env_factory=make_eval_factory(),
        policy=dreamer_planner,
        episodes=cfg_dict["benchmark_episodes"],
        horizon=cfg_dict["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    dreamer_card = compute_scorecard(
        dreamer_results,
        policy_name="tabular-world-model (DreamerV3 dynamics)",
        compute_per_plan_call=dreamer_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(dreamer_card)

    print("[4/4] Computing CPG...")
    cpg = counterfactual_planning_gap(oracle_results, dreamer_results)
    verdict = cpg_verdict(cpg)
    print(f"  oracle    success = {cpg.oracle_success_rate:.3f} (n={cpg.n_episodes_oracle})")
    print(f"  DreamerV3 success = {cpg.learned_success_rate:.3f} (n={cpg.n_episodes_learned})")
    print(f"  CPG = {cpg.gap:+.3f}  95% AC CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]")
    if args.smoke:
        print("  Verdict: SMOKE MODE (config too small for diagnosis; verdict suppressed)")
    else:
        print(f"  Verdict: {verdict}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_cartpole_swingup",
        "metric": "counterfactual_planning_gap",
        "learned_model": "dreamerv3",
        "cpg": {**asdict(cpg), "verdict": verdict},
        "config": cfg_dict,
        "training": {
            "training_steps": cfg_dict["training_steps"],
            "implementation": "dreamerv3-torch (dmc_proprio)",
            "frame_skip": 1,
            "markovian_projection": (
                "recurrent latent truncated to one-frame posterior per dynamics "
                "call; see wmel.adapters.dreamerv3_adapter docstring"
            ),
        },
        "seed": seed,
        "smoke_mode": args.smoke,
        "varied_init": args.varied_init,
        "oracle_scorecard": {
            "policy_name": oracle_card.policy_name,
            "success_rate": oracle_card.success_rate,
            "average_steps_to_success": oracle_card.average_steps_to_success,
            "average_planning_latency_ms": oracle_card.average_planning_latency_ms,
            "average_compute_per_decision": oracle_card.average_compute_per_decision,
            "episodes": oracle_card.episodes,
        },
        "learned_scorecard": {
            "policy_name": dreamer_card.policy_name,
            "success_rate": dreamer_card.success_rate,
            "average_steps_to_success": dreamer_card.average_steps_to_success,
            "average_planning_latency_ms": dreamer_card.average_planning_latency_ms,
            "average_compute_per_decision": dreamer_card.average_compute_per_decision,
            "episodes": dreamer_card.episodes,
        },
        "oracle_full": to_json_report(oracle_results, oracle_card),
        "learned_full": to_json_report(dreamer_results, dreamer_card),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {json_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
