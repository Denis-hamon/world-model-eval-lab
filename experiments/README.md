# experiments/

Training and evaluation scripts that produce reproducible artifacts under [`results/`](../results/). Each subdirectory is named after the (environment, model) pair it targets.

## Convention

Each experiment subdirectory should contain:

- `README.md` — one paragraph stating the question the experiment answers.
- `train.py` (if training is involved) — single-file, reproducible, deterministic with a `--seed` flag.
- `evaluate.py` — runs the trained model through `wmel.benchmark_runner.BenchmarkRunner` and writes a scorecard JSON to the matching `results/<env>_<model>/` directory.
- `requirements.txt` — additional pip dependencies beyond what `pyproject.toml` declares as optional.

## Status

**Currently empty.** This directory is a placeholder for the v0.8+ work outlined in [docs/05_30_day_prototype_plan.md](../docs/05_30_day_prototype_plan.md). The first real experiments will be:

- `dmc_acrobot_gru/` — train a small GRU world model on DMC Acrobot, evaluate via `BenchmarkRunner`, compare against random and oracle dynamics. Produces the first result on a non-toy environment.
- `dmc_acrobot_cpg/` — same env, same model, but reports the **Counterfactual Planning Gap** metric (a candidate new metric defined in v0.8). Quantifies how much the model's prediction errors cost at planning time.

The toy demos under [`examples/`](../examples/) remain the canonical "how to use the framework" entry point; this directory is where the framework gets tested against actual world-model literature.
