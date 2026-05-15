## Summary

<!-- One or two sentences explaining the *why*, not the *what*. -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New metric / benchmark card / adapter / perturbation
- [ ] Documentation only (no code touched)
- [ ] Breaking change (existing callers need to adapt)

## Checklist

- [ ] `pytest -q` passes locally
- [ ] Existing examples in `examples/*/run_*.py` still run from a clean checkout
- [ ] New invariants have a regression test (especially for anything metric-correctness related)
- [ ] Doc and code agree (for a new metric: the definition formula matches the implementation)
- [ ] Commit message explains the *why*, not just the *what*

## If this is a new metric / adapter / perturbation

- [ ] `docs/02_metric_taxonomy.md` / `03_benchmark_cards.md` updated as appropriate
- [ ] Tooltip / popover content updated to match
- [ ] `CONTRIBUTING.md` extension procedure followed
