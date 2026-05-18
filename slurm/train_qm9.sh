#!/bin/bash
# ============================================================
# train_qm9.sh — Self-bootstrapping geo-model training job
#
# Fully portable across any gnode:
#   - Uses /home2 (NFS) for logs, shared data, and checkpoints
#   - Uses /scratch (Node-local) for venv, repo, and fast data I/O
#   - Auto-resumes from NFS checkpoints
#
# Usage:  sbatch slurm/train_qm9.sh
# ============================================================

#SBATCH --job-name=geomodel-qm9
#SBATCH --output=/home2/nishanth.r/geo-model-logs/slurm_%j.log
#SBATCH --error=/home2/nishanth.r/geo-model-logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2

# ── NFS Paths (Shared across all nodes) ────────────────────────────────────
HOME_LOGS="/home2/nishanth.r/geo-model-logs"
SHARED_CKPT="/home2/nishanth.r/geo-model-checkpoints/qm9"
SHARED_DATA="/home2/nishanth.r/geo-model-data/qm9_heavy.jsonl"

# ── Local Paths (Node-specific fast storage) ───────────────────────────────
LOCAL_RUN="/scratch/nishanth.r/geo-model-runs"
REPO_DIR="${LOCAL_RUN}/geo-model"
VENV_DIR="${LOCAL_RUN}/venv_v2"
LOCAL_DATA="${LOCAL_RUN}/data/qm9"

mkdir -p "${HOME_LOGS}" "${SHARED_CKPT}" "${LOCAL_RUN}" "${LOCAL_DATA}"

echo "============================================================"
echo "  geo-model: Self-Bootstrapping QM9 Training"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : $(hostname)"
echo "  Start    : $(date)"
echo "============================================================"

# ── 1. Clone / update repo on local scratch ───────────────────────────────
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[1/5] Cloning repo ..."
    git clone "https://github.com/Nishanth-nishu/geom_confom_gen.git" "${REPO_DIR}"
else
    echo "[1/5] Repo exists — pulling latest..."
    git -C "${REPO_DIR}" fetch origin main
    git -C "${REPO_DIR}" reset --hard origin/main
fi

# ── 2. Create venv on local scratch ────────────────────────────────────────
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[2/5] Creating venv at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip -q
    echo "      Installing PyTorch (CUDA 12.4)..."
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    echo "      Installing PyG..."
    pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html -q
    echo "      Installing other requirements..."
    pip install -r "${REPO_DIR}/requirements.txt" -q
    pip install -e "${REPO_DIR}" -q
    echo "      Venv ready."
else
    source "${VENV_DIR}/bin/activate"
    echo "[2/5] Venv exists — activated."
fi

python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | sed 's/^/  /'

# ── 3. Prepare data on local scratch ───────────────────────────────────────
JSONL="${LOCAL_DATA}/qm9_heavy.jsonl"
if [ ! -f "${JSONL}" ] || [ "$(wc -c < "${JSONL}")" -lt 1000000 ]; then
    echo "[3/5] Copying data from NFS to local scratch..."
    if [ -f "${SHARED_DATA}" ]; then
        cp "${SHARED_DATA}" "${JSONL}"
        echo "      Data copied successfully."
    else
        echo "  ERROR: Shared data not found at ${SHARED_DATA}"
        exit 1
    fi
else
    echo "[3/5] Data ready on local scratch: ${JSONL}"
fi

# ── 4. Patch config to use local data and shared checkpoints ───────────────
echo "[4/5] Configuring paths..."
CONFIG="${LOCAL_RUN}/qm9_run.yaml"
cp "${REPO_DIR}/configs/training/qm9.yaml" "${CONFIG}"
sed -i "s|root:.*|root: ${LOCAL_DATA}|" "${CONFIG}"
sed -i "s|save_dir:.*|save_dir: ${SHARED_CKPT}|" "${CONFIG}"

# ── 5. Train ───────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS=4

LATEST=$(ls -t "${SHARED_CKPT}"/epoch_*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [ -n "${LATEST}" ]; then
    echo "[5/5] Resuming from shared checkpoint: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
else
    echo "[5/5] Starting from scratch."
fi

cd "${REPO_DIR}"
python -u scripts/train.py --config "${CONFIG}" ${RESUME_ARG}

echo ""
echo "Training complete: $(date)"
