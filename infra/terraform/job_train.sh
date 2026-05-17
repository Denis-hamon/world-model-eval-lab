#!/usr/bin/env bash
# Runs inside the OVH AI Training container. The wrapping ovhai call binds
# this script to /workspace and passes it the env vars set in main.tf.
#
# WMEL_GIT_REF   : branch/tag/commit of world-model-eval-lab to check out
# EXPERIMENT_CMD : Python module path (after "python -m") to execute
set -euo pipefail

WMEL_GIT_REF="${WMEL_GIT_REF:-main}"
EXPERIMENT_CMD="${EXPERIMENT_CMD:-experiments.dmc_acrobot.cpg}"

echo "=== environment ==="
echo "hostname     : $(hostname)"
echo "wmel git ref : ${WMEL_GIT_REF}"
echo "experiment   : ${EXPERIMENT_CMD}"
python --version || true
nvidia-smi || echo "(no nvidia-smi — GPU may be unavailable; continuing)"

echo "=== clone wmel ==="
cd /workspace
git clone --depth 1 --branch "${WMEL_GIT_REF}" https://github.com/Denis-hamon/world-model-eval-lab.git wmel
cd wmel

echo "=== install wmel + extras ==="
pip install --no-cache-dir -e ".[dev,control,learned]"

echo "=== sanity checks ==="
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device   :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
python -c "from dm_control import suite; env = suite.load('acrobot', 'swingup'); print('dm-control OK')"

echo "=== run experiment: python -m ${EXPERIMENT_CMD} ==="
python -m "${EXPERIMENT_CMD}"

echo "=== copy results to datastore output mount ==="
# /workspace/output is mounted as a Swift container by ovhai when the job is
# submitted with `--volume <bucket>@<region>:/workspace/output:RW`. The
# terraform main.tf does not currently add this mount automatically — see
# README for the manual one-liner that adds it on a per-run basis.
if [[ -d /workspace/output ]]; then
  cp -rv results/ /workspace/output/ || true
  echo "results written to /workspace/output"
else
  echo "no /workspace/output mount detected; results remain in container FS at $(pwd)/results"
fi

echo "=== done ==="
