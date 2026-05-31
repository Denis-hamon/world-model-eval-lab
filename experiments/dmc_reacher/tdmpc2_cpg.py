"""Train TD-MPC2 on DMC Reacher-easy, then run the CPG protocol.

Third environment after Acrobot and Cartpole, and the first with a 2-D
action (a 3x3 = 9-action discrete torque grid over the two joints). It
extends the v0.11 worked example (random-shooting planner + bespoke MLP)
by replacing the learned dynamics with a TD-MPC2 latent world model, on a
2-DOF reaching task rather than an underactuated swing-up.

The CPG arm structure mirrors `experiments.dmc_cartpole.tdmpc2_cpg`:
same `BenchmarkRunner`, same `TabularWorldModelPlanner`, same
`reacher_reach_score`, same `make_reacher_oracle_dynamics`. Only the
learned dynamics callable changes.

Pipeline
--------
1. Train TD-MPC2 on DMC Reacher-easy for `--steps` env steps. The
   DMControl wrapper is patched to step once per agent action (TD-MPC2's
   default wrapper does 2-step action-repeat; wmel's env is 1-step, and
   the planner needs a 1-step learned dynamics for a like-for-like
   comparison with the oracle).
2. Save the trained TD-MPC2 agent. Resume from this checkpoint if the
   script is rerun.
3. Collect a fresh on-policy rollout (eval-mode TD-MPC2 actions), encode
   each observation, and fit a small decoder z -> obs on the (z, obs)
   pairs. TD-MPC2 ships no obs decoder; the planner contract is in obs
   space; this decoder bridges the two.
4. Save the (encoder, dynamics, decoder) triplet in
   `wmel.adapters.tdmpc2_adapter` format.
5. Run the CPG benchmark twice: oracle dynamics arm, TD-MPC2 dynamics
   arm. Dump a versioned JSON report.

Setup
-----
    ./scripts/setup_tdmpc2.sh                # clones TD-MPC2 to third_party/
    pip install -e ".[dev,control,learned]"  # core wmel deps
    pip install torchrl==0.6.0 --no-deps     # TD-MPC2 buffer/agent deps
    pip install tensordict hydra-core omegaconf gymnasium==0.29.1 \
                termcolor pandas h5py kornia

Usage
-----
    python -m experiments.dmc_reacher.tdmpc2_cpg --smoke         # ~5 min
    python -m experiments.dmc_reacher.tdmpc2_cpg                 # 500k steps, ~12-20h
    python -m experiments.dmc_reacher.tdmpc2_cpg --steps 100000  # custom

Smoke mode is the contract test: end-to-end pipeline at small scale, used
to validate the wiring before committing GPU hours.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Headless MuJoCo: must be set before any dm_control import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LAZY_LEGACY_OP", "0")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TDMPC2_PKG = _REPO_ROOT / "third_party" / "tdmpc2" / "tdmpc2"
for _entry in (_REPO_ROOT, _REPO_ROOT / "src", _TDMPC2_PKG):
    s = str(_entry)
    if _entry.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

import numpy as np
import torch
from omegaconf import OmegaConf

import hydra.utils
hydra.utils.get_original_cwd = lambda: os.getcwd()  # parse_cfg uses this for work_dir

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env as tdmpc2_make_env
from envs import dmcontrol as tdmpc2_dmcontrol
from tdmpc2 import TDMPC2
from tensordict.tensordict import TensorDict

from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.adapters.tdmpc2_adapter import TDMPC2Dynamics, make_tdmpc2_dynamics
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_reacher import DMCReacherEnv, reacher_reach_score, make_reacher_oracle_dynamics
from wmel.metrics import compute_scorecard, counterfactual_planning_gap, cpg_verdict
from wmel.report import print_scorecard, report_envelope_metadata, to_json_report


CHECKPOINT_PATH = _REPO_ROOT / "results" / "dmc_reacher" / "tdmpc2_reacher.pt"
JSON_PATH = _REPO_ROOT / "results" / "dmc_reacher" / "tdmpc2_cpg.json"
TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_reacher" / "tdmpc2_agent.pt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true", help="Smaller config for end-to-end validation (~5 min).")
    p.add_argument("--steps", type=int, default=None, help="Override training steps (default: 500_000, smoke: 5_000).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-size", type=int, default=1, help="TD-MPC2 size preset (1, 5, 19, ...). Sets cfg.model_size.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _output_suffix(model_size: int, seed: int) -> str:
    # size=1, seed=0 keeps the legacy unsuffixed names; size=1 with seed>0
    # uses the phase-5m-cartpole-seed{N} convention; size!=1 always
    # carries both axes to avoid collisions across (size, seed) pairs.
    if model_size == 1 and seed == 0:
        return ""
    if model_size == 1:
        return f"_seed{seed}"
    return f"_size{model_size}_seed{seed}"


def _patch_dmcontrol_no_frame_skip() -> None:
    """Make TD-MPC2's DMControlWrapper step once per agent action.

    The default 2-step action-repeat would make the learned dynamics
    predict 2 dmc-steps ahead, while wmel's oracle and env both run at
    1 dmc-step per call. Matching the timescale is required for a
    like-for-like CPG.
    """
    cls = tdmpc2_dmcontrol.DMControlWrapper
    original_step = cls.step

    def step_once(self, action):
        action = action.astype(self.action_spec_dtype)
        step = self.env.step(action)
        from collections import defaultdict
        return self._obs_to_array(step.observation), float(step.reward or 0.0), False, defaultdict(float)

    cls.step = step_once
    # tag for the report so a future reader can tell what they're looking at
    cls._wmel_frame_skip = 1


def _build_cfg(task: str, steps: int, seed: int, model_size: int = 1) -> object:
    cfg = OmegaConf.load(str(_TDMPC2_PKG / "config.yaml"))
    cfg.task = task
    cfg.obs = "state"
    cfg.steps = int(steps)
    cfg.seed = int(seed)
    cfg.model_size = int(model_size)
    cfg.compile = False           # avoid torch.compile fragility on this stack
    cfg.save_video = False
    cfg.save_agent = False
    cfg.save_csv = False
    cfg.enable_wandb = False
    cfg.exp_name = "tdmpc2_reacher"
    cfg.data_dir = "/tmp/tdmpc2_data"
    cfg.eval_freq = 10_000_000    # skip in-training eval; we evaluate via wmel CPG
    cfg.eval_episodes = 0
    cfg.buffer_size = 100_000
    cfg.work_dir = str(_REPO_ROOT / "results" / "dmc_reacher" / "tdmpc2_workdir")
    cfg.wandb_project = "wmel"
    cfg.wandb_entity = "noop"
    cfg = parse_cfg(cfg)
    return cfg


def _make_buffer_td(obs, action, reward, terminated, rand_action_template) -> TensorDict:
    if action is None:
        action = torch.full_like(rand_action_template, float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan"))
    if terminated is None:
        terminated = torch.tensor(float("nan"))
    return TensorDict(
        obs=obs.unsqueeze(0).cpu(),
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
        terminated=terminated.unsqueeze(0),
        batch_size=(1,),
    )


def _train_tdmpc2(cfg, env, device: str, resume_from: Path | None) -> TDMPC2:
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)

    start_step = 0
    if resume_from is not None and resume_from.exists():
        state = torch.load(resume_from, map_location=device, weights_only=False)
        agent.model.load_state_dict(state["model_state"])
        start_step = int(state.get("step", 0))
        print(f"[resume] loaded agent state from {resume_from.name} at step {start_step}")

    rand_act_template = env.rand_act()
    print(f"[train] starting at step {start_step}, target {cfg.steps}, seed_steps={cfg.seed_steps}")

    obs = env.reset()
    tds: list[TensorDict] = [_make_buffer_td(obs, None, None, None, rand_act_template)]
    ep_return = 0.0
    ep_len = 0
    last_log_t = time.time()
    last_log_step = start_step
    pretrained_on_seed = (start_step > cfg.seed_steps)

    step = start_step
    while step <= cfg.steps:
        if step > cfg.seed_steps:
            action = agent.act(obs, t0=(len(tds) == 1))
        else:
            action = env.rand_act()

        obs, reward, done, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1
        tds.append(_make_buffer_td(obs, action, torch.tensor(float(reward)), torch.tensor(float(info.get("terminated", 0.0))), rand_act_template))

        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if (step == cfg.seed_steps and not pretrained_on_seed) else 1
            if num_updates > 1:
                print(f"[train] pretraining agent on {num_updates} seed updates...")
                pretrained_on_seed = True
            for _ in range(num_updates):
                agent.update(buffer)

        step += 1

        if done or ep_len >= cfg.episode_length:
            buffer.add(torch.cat(tds))
            now = time.time()
            sps = (step - last_log_step) / max(now - last_log_t, 1e-6)
            print(f"[train] step={step:>7d}  ep_return={ep_return:7.2f}  ep_len={ep_len:>3d}  sps={sps:6.1f}")
            last_log_t, last_log_step = now, step
            obs = env.reset()
            tds = [_make_buffer_td(obs, None, None, None, rand_act_template)]
            ep_return = 0.0
            ep_len = 0

            if step % 20_000 == 0 or step >= cfg.steps:
                _save_agent(agent, step)

    _save_agent(agent, step)
    return agent


def _save_agent(agent: TDMPC2, step: int) -> None:
    TDMPC2_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": agent.model.state_dict(), "step": step}, TDMPC2_AGENT_PATH)


def _collect_rollouts_for_decoder(agent: TDMPC2, env, n_episodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eval-mode TD-MPC2 episodes; emit (obs, z) pairs from BOTH
    encoder(obs) AND dynamics(encoder(obs_prev), a). The decoder is used at
    planner time on `dynamics(...)` outputs, which can drift off the encoder
    output manifold; including post-dynamics pairs in its training data
    keeps it in-distribution along the rollout horizon.
    """
    all_obs: list[torch.Tensor] = []
    all_z: list[torch.Tensor] = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0
        prev_z: torch.Tensor | None = None
        prev_action_t: torch.Tensor | None = None
        while not done:
            with torch.no_grad():
                z_enc = agent.model.encode(obs.to(agent.device).unsqueeze(0), task=None)
                all_obs.append(obs.cpu())
                all_z.append(z_enc.cpu().squeeze(0))
                if prev_z is not None and prev_action_t is not None:
                    z_dyn = agent.model.next(prev_z, prev_action_t, task=None)
                    all_obs.append(obs.cpu())
                    all_z.append(z_dyn.cpu().squeeze(0))
                prev_z = z_enc
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            prev_action_t = action.to(agent.device).unsqueeze(0).float()
            obs, _r, done, _info = env.step(action)
            t += 1
            if t >= 500:
                break
    return torch.stack(all_obs), torch.stack(all_z)


def _train_decoder(model: TDMPC2Dynamics, z_batch: torch.Tensor, obs_batch: torch.Tensor, device: str, steps: int) -> dict:
    z_dev = z_batch.to(device)
    o_dev = obs_batch.to(device)
    opt = torch.optim.Adam(model.decoder.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    bs = min(256, len(z_dev))
    n = len(z_dev)
    final_loss = float("nan")
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        pred = model.decoder(z_dev[idx])
        loss = loss_fn(pred, o_dev[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        if s == 0 or (s + 1) % max(1, steps // 5) == 0:
            print(f"[decoder] step {s+1:>5d}/{steps}  mse={final_loss:.6f}")
    with torch.no_grad():
        full_mse = float(loss_fn(model.decoder(z_dev), o_dev).item())
    return {"final_train_mse": final_loss, "full_dataset_mse": full_mse, "samples": n, "steps": steps}


def _extract_arch(cfg) -> dict:
    return {
        "obs_dim": int(cfg.obs_shape["state"][0]),
        "action_dim": int(cfg.action_dim),
        "latent_dim": int(cfg.latent_dim),
        "enc_dim": int(cfg.enc_dim),
        "mlp_dim": int(cfg.mlp_dim),
        "num_enc_layers": int(cfg.num_enc_layers),
        "simnorm_dim": int(cfg.simnorm_dim),
        "decoder_hidden": 256,
    }


def _port_tdmpc2_to_adapter(agent: TDMPC2, arch: dict) -> TDMPC2Dynamics:
    """Copy encoder + dynamics weights from a TD-MPC2 agent into a fresh
    TDMPC2Dynamics. The decoder is left at init (random) and will be
    trained separately on rollout data.
    """
    model = TDMPC2Dynamics(**arch)
    src = agent.model.state_dict()
    enc_prefix = "_encoder.state."
    dyn_prefix = "_dynamics."
    enc_dst = {k[len(enc_prefix):]: v for k, v in src.items() if k.startswith(enc_prefix)}
    dyn_dst = {k[len(dyn_prefix):]: v for k, v in src.items() if k.startswith(dyn_prefix)}
    missing_enc, unexpected_enc = model.encoder.load_state_dict(enc_dst, strict=False)
    missing_dyn, unexpected_dyn = model.dynamics.load_state_dict(dyn_dst, strict=False)
    if missing_enc or unexpected_enc:
        raise RuntimeError(f"encoder weight mismatch: missing={missing_enc} unexpected={unexpected_enc}")
    if missing_dyn or unexpected_dyn:
        raise RuntimeError(f"dynamics weight mismatch: missing={missing_dyn} unexpected={unexpected_dyn}")
    return model


def _config(smoke: bool, steps_override: int | None) -> dict:
    if smoke:
        return {
            "training_steps": steps_override if steps_override is not None else 5_000,
            "decoder_rollout_episodes": 2,
            "decoder_train_steps": 500,
            "num_candidates": 15,
            "plan_horizon": 8,
            "benchmark_episodes": 2,
            "benchmark_horizon": 80,
        }
    return {
        "training_steps": steps_override if steps_override is not None else 500_000,
        "decoder_rollout_episodes": 10,
        "decoder_train_steps": 5_000,
        "num_candidates": 50,
        "plan_horizon": 15,
        "benchmark_episodes": 10,
        "benchmark_horizon": 500,
    }


def main() -> None:
    args = _parse_args()
    cfg_dict = _config(smoke=args.smoke, steps_override=args.steps)
    seed = args.seed
    model_size = args.model_size
    device = args.device if torch.cuda.is_available() else "cpu"
    suffix = _output_suffix(model_size, seed)
    global CHECKPOINT_PATH, JSON_PATH, TDMPC2_AGENT_PATH
    CHECKPOINT_PATH = _REPO_ROOT / "results" / "dmc_reacher" / f"tdmpc2_reacher{suffix}.pt"
    JSON_PATH = _REPO_ROOT / "results" / "dmc_reacher" / f"tdmpc2_cpg{suffix}.json"
    TDMPC2_AGENT_PATH = _REPO_ROOT / "results" / "dmc_reacher" / f"tdmpc2_agent{suffix}.pt"
    print(f"[setup] device={device}, smoke={args.smoke}, training_steps={cfg_dict['training_steps']}, seed={seed}, model_size={model_size}")

    _patch_dmcontrol_no_frame_skip()
    tdmpc2_cfg = _build_cfg(task="reacher-easy", steps=cfg_dict["training_steps"], seed=seed, model_size=model_size)
    set_seed(seed)

    print("[1/5] Building TD-MPC2 env and agent...")
    env = tdmpc2_make_env(tdmpc2_cfg)

    print("[2/5] Training TD-MPC2...")
    agent = _train_tdmpc2(tdmpc2_cfg, env, device=device, resume_from=TDMPC2_AGENT_PATH)

    print("[3/5] Collecting on-policy rollouts and training decoder...")
    obs_batch, z_batch = _collect_rollouts_for_decoder(agent, env, n_episodes=cfg_dict["decoder_rollout_episodes"])
    arch = _extract_arch(tdmpc2_cfg)
    adapter_model = _port_tdmpc2_to_adapter(agent, arch).to(device)
    decoder_log = _train_decoder(adapter_model, z_batch, obs_batch, device=device, steps=cfg_dict["decoder_train_steps"])

    # Reacher's action is 2-D: store the explicit 9-action grid (the 3x3
    # Cartesian product of per-joint torque levels) so make_tdmpc2_dynamics
    # rebuilds the same discrete set. The 1-D `action_levels` key is for the
    # swing-up envs only.
    env_template = DMCReacherEnv()
    action_set = [list(a) for a in env_template.action_space]
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in adapter_model.state_dict().items()},
            "arch": arch,
            "action_set": action_set,
            "meta": {
                "training_steps": cfg_dict["training_steps"],
                "seed": seed,
                "decoder": decoder_log,
                "frame_skip": 1,
                "tdmpc2_model_size": int(tdmpc2_cfg.model_size),
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"[ckpt] wrote {CHECKPOINT_PATH.relative_to(_REPO_ROOT)}  (decoder mse={decoder_log['full_dataset_mse']:.6f})")

    print("[4/5] Building wmel arms (oracle + TD-MPC2 dynamics) and running benchmarks...")
    def make_planner(dyn):
        return TabularWorldModelPlanner(
            dynamics=dyn,
            action_space=env_template.action_space,
            num_candidates=cfg_dict["num_candidates"],
            plan_horizon=cfg_dict["plan_horizon"],
            score=reacher_reach_score,
            seed=seed,
        )

    oracle_planner = make_planner(make_reacher_oracle_dynamics())
    oracle_results = BenchmarkRunner(
        env_factory=lambda: DMCReacherEnv(),
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

    tdmpc2_dyn = make_tdmpc2_dynamics(CHECKPOINT_PATH, device="cpu")
    tdmpc2_planner = make_planner(tdmpc2_dyn)
    tdmpc2_results = BenchmarkRunner(
        env_factory=lambda: DMCReacherEnv(),
        policy=tdmpc2_planner,
        episodes=cfg_dict["benchmark_episodes"],
        horizon=cfg_dict["benchmark_horizon"],
        perturb_prob=0.0,
        seed=seed,
    ).run()
    tdmpc2_card = compute_scorecard(
        tdmpc2_results,
        policy_name="tabular-world-model (TD-MPC2 dynamics)",
        compute_per_plan_call=tdmpc2_planner.compute_per_plan_call,
        perturbation_name="env-default",
    )
    print_scorecard(tdmpc2_card)

    print("[5/5] Computing CPG...")
    cpg = counterfactual_planning_gap(oracle_results, tdmpc2_results)
    verdict = cpg_verdict(cpg)
    print(f"  oracle  success = {cpg.oracle_success_rate:.3f} (n={cpg.n_episodes_oracle})")
    print(f"  TD-MPC2 success = {cpg.learned_success_rate:.3f} (n={cpg.n_episodes_learned})")
    print(f"  CPG = {cpg.gap:+.3f}  95% AC CI [{cpg.gap_ci_low:+.3f}, {cpg.gap_ci_high:+.3f}]")
    if args.smoke:
        print("  Verdict: SMOKE MODE (config too small for diagnosis; verdict suppressed)")
    else:
        print(f"  Verdict: {verdict}")

    report = {
        **report_envelope_metadata(),
        "environment": "dmc_reacher_easy",
        "metric": "counterfactual_planning_gap",
        "learned_model": "tdmpc2",
        "cpg": {**asdict(cpg), "verdict": verdict},
        "config": cfg_dict,
        "training": {
            "training_steps": cfg_dict["training_steps"],
            "tdmpc2_model_size": int(tdmpc2_cfg.model_size),
            "frame_skip": 1,
            "decoder": decoder_log,
        },
        "seed": seed,
        "smoke_mode": args.smoke,
        "oracle_scorecard": {
            "policy_name": oracle_card.policy_name,
            "success_rate": oracle_card.success_rate,
            "average_steps_to_success": oracle_card.average_steps_to_success,
            "average_planning_latency_ms": oracle_card.average_planning_latency_ms,
            "average_compute_per_decision": oracle_card.average_compute_per_decision,
            "episodes": oracle_card.episodes,
        },
        "learned_scorecard": {
            "policy_name": tdmpc2_card.policy_name,
            "success_rate": tdmpc2_card.success_rate,
            "average_steps_to_success": tdmpc2_card.average_steps_to_success,
            "average_planning_latency_ms": tdmpc2_card.average_planning_latency_ms,
            "average_compute_per_decision": tdmpc2_card.average_compute_per_decision,
            "episodes": tdmpc2_card.episodes,
        },
        "oracle_full": to_json_report(oracle_results, oracle_card),
        "learned_full": to_json_report(tdmpc2_results, tdmpc2_card),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {JSON_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
