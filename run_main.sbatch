#!/bin/bash
#SBATCH --job-name=detr_forward_hook
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=csci_ua_0480_042-2025fa
# #SBATCH --qos=normal

set -euo pipefail

NETID="bjd8427"
WRKDIR="$PWD"
DRIVE_DIR="/scratch/${NETID}/DETR-Forward-Hook"
PYFILE="main.py"

SIF="/scratch/${NETID}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
OVERLAY="/scratch/${NETID}/overlay-25GB-500K.ext3:ro"
BIND="/scratch"
SING="singularity exec --nv --bind ${BIND} --overlay ${OVERLAY} ${SIF}"

echo "============================================================"
echo "Job: ${SLURM_JOB_NAME:-n/a} | ID: ${SLURM_JOB_ID:-n/a} | Node: $(hostname) | $(date)"
echo "Workdir: $WRKDIR"
echo "============================================================"

# GPU info
$SING bash -lc 'nvidia-smi || echo "No GPU info available."'

# Host-side caches
mkdir -p "$DRIVE_DIR" "$DRIVE_DIR/.torch-cache" "$DRIVE_DIR/.hf-cache"
mkdir -p "$WRKDIR/logs"

# ---------- Sanity & activation ----------
$SING bash -lc '
  set -e -o pipefail

  export MKL_INTERFACE_LAYER=LP64
  export MKL_THREADING_LAYER=GNU

  if [ -f /ext3/env.sh ]; then
    source /ext3/env.sh
  else
    echo "ERROR: /ext3/env.sh not found inside container."; exit 1
  fi

  set +u
  conda activate computer_vision || { echo "ERROR: conda activate failed"; exit 1; }
  set -u

  echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
  which python; python -V

  python - <<PY
import sys
try:
    import torch, torchvision
    print("✅ PyTorch:", torch.__version__, "| CUDA avail:", getattr(__import__("torch"), "cuda").is_available())
except Exception as e:
    print("❌ PyTorch import failed:", e); sys.exit(1)
PY
'

# ---------- Run ----------
echo "============================================================"
echo "Running inside container: python $PYFILE"
echo "============================================================"

$SING bash -lc "
  set -e -o pipefail
  export MKL_INTERFACE_LAYER=LP64
  export MKL_THREADING_LAYER=GNU

  source /ext3/env.sh
  set +u
  conda activate computer_vision
  set -u

  export MPLBACKEND=Agg
  export PYTHONUNBUFFERED=1
  export TORCH_HOME='$DRIVE_DIR/.torch-cache'
  export TRANSFORMERS_CACHE='$DRIVE_DIR/.hf-cache'

  cd '$WRKDIR'
  python '$PYFILE'
"

echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
