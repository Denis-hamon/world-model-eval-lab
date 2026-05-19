#!/usr/bin/env bash
# Clone TD-MPC2 source into third_party/ for the
# `experiments.dmc_acrobot.tdmpc2_cpg` experiment.
#
# Why we vendor it this way: TD-MPC2's internal imports are bare
# (`from common import ...`), which only works when its own package dir is on
# sys.path. The experiment script adds `third_party/tdmpc2/tdmpc2` to
# sys.path at runtime, so we just need the source tree at that path. We do
# not pip-install TD-MPC2 to keep the wmel package itself free of
# tdmpc2-specific deps (hydra, gymnasium, ...).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/third_party/tdmpc2"
mkdir -p "$REPO_ROOT/third_party"
if [ -d "$DEST/.git" ]; then
    echo "TD-MPC2 already present at $DEST"
    exit 0
fi
git clone --depth 1 https://github.com/nicklashansen/tdmpc2.git "$DEST"
echo "TD-MPC2 cloned to $DEST"
