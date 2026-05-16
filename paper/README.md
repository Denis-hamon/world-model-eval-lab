# paper/

Short paper that accompanies the framework: **"Counterfactual Planning Gap: A Decision-Grade Metric for Decoupling Model Error from Planner Capacity in World Model Evaluation."**

The source is plain LaTeX. The paper is ~7 pages including references and is intended for an arXiv preprint or an ML workshop submission. The body matches what the repository's code, tests, and committed results actually deliver - no claims appear in the paper that are not reproducible from a fresh checkout.

## Contents

- [`main.tex`](main.tex) - the paper. Standard `article` class, no exotic packages. Sections: introduction; related work; the evaluation contract and decision-grade taxonomy; the CPG metric definition, statistics, and gated verdict; an empirical study on DMC Acrobot-swingup; discussion / limitations.
- [`references.bib`](references.bib) - 23 entries. Latent-dynamics world models (Dreamer-V3, MuZero, IRIS, Genie, PlaNet), JEPA-line predictive architectures (LeCun 2022, I-JEPA, V-JEPA, V-JEPA 2), model exploitation in offline MBRL (MOPO, MOReL), evaluation methodology (Wilson 1927, Agresti-Caffo 2000, Newcombe 1998, Henderson et al. 2018, rliable), benchmarks (DM Control Suite, OGBench, LIBERO).
- [`Makefile`](Makefile) - `make` builds `main.pdf` via `pdflatex` + `bibtex`. Requires `texlive-latex-extra` and `texlive-bibtex-extra`.
- [`build_figures.py`](build_figures.py) - prints the numeric values that populate the headline table in `main.tex`, sourced directly from `results/dmc_acrobot/cpg.json`. Lets a reader confirm the paper's numbers match the latest committed scorecard.

## Reproducibility

Every number in the paper's headline table comes from `results/dmc_acrobot/cpg.json`, which is itself produced by `python -m experiments.dmc_acrobot.cpg` from a clean checkout. To regenerate:

```bash
pip install -e ".[dev,control,learned]"
python -m experiments.dmc_acrobot.cpg
python paper/build_figures.py   # prints the LaTeX-ready values
```

Then update the table in `main.tex` if anything drifts (with seed=0 it should not).

## What the paper claims, honestly

The empirical contribution is intentionally narrow: one environment, one learned-model architecture, one planner, one seed, ten episodes per arm. The verdict at that sample size is **INCONCLUSIVE** -- the raw CPG is $+0.30$ but the Agresti--Caffo 95\% CI on the gap crosses zero. The paper argues that a metric returning the honest "inconclusive" at small $n$ is the correctly-calibrated behaviour and outlines what a multi-seed extension would change.

The methodological contributions -- the CPG definition with AC CI, the gated verdict, and the four-method evaluation contract that makes computing CPG cheap -- are independent of any specific empirical result.

## Build status

The paper has not been pre-compiled on the machine that ships this commit (no LaTeX toolchain was available). The source is standard enough that it should compile on any TeX Live 2020+ distribution. A future CI workflow can publish `main.pdf` as a release artifact.

## Non-affiliation

This paper is independent. It is not an official artifact of the AMI program at Meta, the LeWorldModel project, the authors of any cited paper, or any current or past employer of the author. JEPA-style and LeWorldModel references are conceptual, not affiliational.
