#!/usr/bin/env bash
# Clone the reference PyTorch DreamerV3 implementation (NM512/dreamerv3-torch)
# into third_party/ for the `experiments.dmc_acrobot.dreamerv3_cpg` experiment.
#
# Why we vendor it this way: the experiment trains DreamerV3 by invoking
# upstream's dreamer.py as a subprocess (never importing it), then ports the
# world-model weights into `wmel.adapters.dreamerv3_adapter`. We do not
# pip-install dreamerv3-torch (it is not packaged) and we keep its training
# deps out of the wmel package. Upstream pins old mujoco/dm_control versions
# in its requirements.txt; if those conflict with the wmel [control] extra in
# your environment, train in a separate venv and hand the latest.pt to the
# experiment via --agent-checkpoint.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/third_party/dreamerv3-torch"
mkdir -p "$REPO_ROOT/third_party"
if [ -d "$DEST/.git" ]; then
    echo "dreamerv3-torch already present at $DEST"
    exit 0
fi
git clone --depth 1 https://github.com/NM512/dreamerv3-torch.git "$DEST"
echo "dreamerv3-torch cloned to $DEST"
