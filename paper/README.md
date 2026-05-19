# paper/

Short paper that accompanies the framework: **"Counterfactual Planning Gap: A Decision-Grade Metric for Decoupling Model Error from Planner Capacity in World Model Evaluation."**

The source is plain LaTeX. The paper is ~7 pages including references and is intended for an arXiv preprint or an ML workshop submission. The body matches what the repository's code, tests, and committed results actually deliver - no claims appear in the paper that are not reproducible from a fresh checkout.

## Contents

- [`main.tex`](main.tex) - the paper. Standard `article` class, no exotic packages. Sections: introduction; related work; the evaluation contract and decision-grade taxonomy; the CPG metric definition, statistics, and gated verdict; an empirical study on DMC Acrobot-swingup including a multi-seed sweep, a coverage-axis receipt, and a robustness check that adds a published-world-model arm (TD-MPC2) and a CEM planner; discussion / limitations.
- [`references.bib`](references.bib) - 25 entries. Latent-dynamics world models (Dreamer-V3, MuZero, IRIS, Genie, PlaNet, TD-MPC2), JEPA-line predictive architectures (LeCun 2022, I-JEPA, V-JEPA, V-JEPA 2), model exploitation in offline MBRL (MOPO, MOReL), evaluation methodology (Wilson 1927, Agresti-Caffo 2000, Newcombe 1998, Henderson et al. 2018, rliable), benchmarks (DM Control Suite, OGBench, LIBERO).
- [`Makefile`](Makefile) - `make` builds `main.pdf` via `pdflatex` + `bibtex`. Requires `texlive-latex-extra` and `texlive-bibtex-extra`.
- [`build_figures.py`](build_figures.py) - prints the numeric values that populate the headline (single-seed) CPG table in `main.tex`, sourced from `results/dmc_acrobot/cpg.json`. The multi-seed-sweep table and the robustness table are populated from `cpg_sweep.json`, `tdmpc2_cpg.json`, `coverage_mlp_on_tdmpc2_cpg.json`, and `cem_cpg.json`; those numbers were transcribed by hand at write time and can be re-extracted with `python -c 'import json; print(json.load(open("results/dmc_acrobot/<file>.json")))'`.

## Reproducibility

Every number in the paper comes from a committed JSON under `results/dmc_acrobot/`:

```bash
pip install -e ".[dev,control,learned]"
python -m experiments.dmc_acrobot.cpg                   # cpg.json (Table 1)
python -m experiments.dmc_acrobot.cpg_sweep             # cpg_sweep.json (Table 2)
./scripts/setup_tdmpc2.sh                               # clones TD-MPC2 to third_party/
# Robustness rows (Table 3) need GPU + the TD-MPC2 dependency stack:
python -m experiments.dmc_acrobot.tdmpc2_cpg            # ~12-20 h on an RTX 5000, model_size=1
python -m experiments.dmc_acrobot.coverage_mlp_on_tdmpc2 # ~30 min after the agent ckpt exists
python -m experiments.dmc_acrobot.cem_cpg               # ~1 h after the agent ckpt exists
python paper/build_figures.py                           # prints LaTeX-ready values for Table 1
```

With `seed=0` the numbers in the paper should match exactly. The full TD-MPC2 training run is the only step that needs a GPU; everything else is CPU-only.

## What the paper claims, honestly

The empirical contribution is intentionally narrow: one environment. The paper reports the headline CPG at $n = 10$ as **INCONCLUSIVE** (CPG $= +0.300$, AC CI $[-0.06, +0.56]$), hardens it to **MODEL BOTTLENECK** at pooled-150 with a $\sim 150\times$ sweep of training-set size, and then runs a robustness check that swaps the bespoke MLP for TD-MPC2 trained on $2 \times 10^{6}$ env steps and the random-shooting planner for a CEM planner. Both learned arms remain at $0/10$ across both planners; the oracle's success rate triples under CEM ($0.30 \to 0.90$); the verdict opens to **MODEL BOTTLENECK** at $n = 10$ with CPG $= +0.900$, CI $[+0.49, +1.01]$. The methodological argument is that this decomposition -- a dynamics-quality bottleneck on the learned arms separable from a planner-capacity contributor on the oracle arm -- is what a prediction-quality metric alone cannot surface.

The methodological contributions -- the CPG definition with AC CI, the gated verdict, and the four-method evaluation contract that makes computing CPG cheap -- are independent of any specific empirical result.

## Build status

The paper has not been pre-compiled on the machine that ships this commit (no LaTeX toolchain was available). The source is standard enough that it should compile on any TeX Live 2020+ distribution. A future CI workflow can publish `main.pdf` as a release artifact.

## Non-affiliation

This paper is independent. It is not an official artifact of the AMI program at Meta, the LeWorldModel project, the authors of any cited paper, or any current or past employer of the author. JEPA-style and LeWorldModel references are conceptual, not affiliational.
