#!/bin/bash
# ============================================================
# train_qm9.sh — Self-bootstrapping geo-model training job
#
# Fully portable: clones repo, builds venv, prepares data,
# and trains — on ANY gnode, with no pre-existing setup.
#
# Usage:  sbatch slurm/train_qm9.sh
# Monitor: tail -f ~/geo-model-logs/slurm_<JOBID>.log
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

# ── Paths ──────────────────────────────────────────────────────────────────
GIT_REPO="https://github.com/Nishanth-nishu/geom_confom_gen.git"
HOME_LOGS="/home2/nishanth.r/geo-model-logs"
SHARED_RUN="/scratch/nishanth.r/geo-model-runs"       # Shared NFS scratch
REPO_DIR="${SHARED_RUN}/geo-model"
VENV_DIR="${SHARED_RUN}/venv"
CKPT_DIR="${SHARED_RUN}/checkpoints/qm9"
DATA_DIR="${SHARED_RUN}/data/qm9"

# GeoDiff source PKLs (on shared /scratch — used to generate data if needed)
GEODIFF_PKL_DIR="/scratch/nishanth.r/nextmol_experiment/GeoDiff/data/GEOM/QM9"

mkdir -p "${HOME_LOGS}" "${SHARED_RUN}" "${CKPT_DIR}" "${DATA_DIR}"

echo "============================================================"
echo "  geo-model: Self-Bootstrapping QM9 Training"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : $(hostname)"
echo "  Run dir  : ${SHARED_RUN}"
echo "  Start    : $(date)"
echo "============================================================"

# ── 1. Clone / update repo ─────────────────────────────────────────────────
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[1/5] Cloning ${GIT_REPO} ..."
    git clone "${GIT_REPO}" "${REPO_DIR}"
else
    echo "[1/5] Repo exists — pulling latest..."
    git -C "${REPO_DIR}" pull --ff-only origin main 2>/dev/null || echo "  (skipping pull, working tree dirty)"
fi

# ── 2. Create venv if needed ───────────────────────────────────────────────
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

# ── 3. Prepare data if needed ──────────────────────────────────────────────
JSONL="${DATA_DIR}/qm9_heavy.jsonl"
if [ ! -f "${JSONL}" ] || [ "$(wc -c < "${JSONL}")" -lt 1000000 ]; then
    echo "[3/5] Data not found — converting from GeoDiff PKLs ..."
    if [ -d "${GEODIFF_PKL_DIR}" ]; then
        # Override output path in the converter
        PYTHONPATH="${REPO_DIR}" python - <<PYEOF
import json, os, pickle, sys
GEODIFF_DATA = "${GEODIFF_PKL_DIR}"
OUT_FILE     = "${JSONL}"
VALID_Z      = {6, 7, 8, 9, 16, 17}
PKL_FILES    = ["train_data_40k.pkl", "val_data_5k.pkl", "test_data_1k.pkl"]

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
n_ok = 0
with open(OUT_FILE, "w") as f_out:
    for pkl in PKL_FILES:
        print(f"  Loading {pkl}...", flush=True)
        with open(os.path.join(GEODIFF_DATA, pkl), "rb") as f:
            dataset = pickle.load(f)
        for item in dataset:
            try:
                atom_type  = item.atom_type.tolist()
                pos        = item.pos.tolist()
                edge_index = item.edge_index.tolist()
                edge_type  = item.edge_type.tolist()
            except Exception:
                continue
            if any(z not in VALID_Z for z in atom_type): continue
            heavy = [z != 1 for z in atom_type]
            old2new, at2, co2 = {}, [], []
            for i, (keep, z, xyz) in enumerate(zip(heavy, atom_type, pos)):
                if keep:
                    old2new[i] = len(at2)
                    at2.append(z)
                    co2.append([round(c, 6) for c in xyz])
            src, dst, bt = [], [], []
            for k in range(len(edge_index[0])):
                i, j = edge_index[0][k], edge_index[1][k]
                if i in old2new and j in old2new:
                    src.append(old2new[i]); dst.append(old2new[j]); bt.append(edge_type[k])
            f_out.write(json.dumps({"atom_types": at2, "edge_index": [src, dst], "bond_types": bt, "coordinates": co2}) + "\n")
            n_ok += 1
        del dataset
        print(f"  {n_ok} molecules so far.", flush=True)
print(f"Done: {n_ok} molecules → {OUT_FILE}")
PYEOF
    else
        echo "  ERROR: GeoDiff PKLs not found at ${GEODIFF_PKL_DIR}"
        echo "  If running on a node without /scratch, you need to either:"
        echo "    a) Run on a node where /scratch is mounted, OR"
        echo "    b) Copy qm9_heavy.jsonl to ${DATA_DIR}/ manually first."
        exit 1
    fi
else
    echo "[3/5] Data ready: ${JSONL}"
fi

# ── 4. Patch config to use shared paths ───────────────────────────────────
echo "[4/5] Configuring paths..."
CONFIG="${SHARED_RUN}/qm9_run.yaml"
cp "${REPO_DIR}/configs/training/qm9.yaml" "${CONFIG}"
# Override data root and checkpoint dir
sed -i "s|root:.*|root: ${DATA_DIR}|" "${CONFIG}"
sed -i "s|save_dir:.*|save_dir: ${CKPT_DIR}|" "${CONFIG}"

# ── 5. Train ───────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONPATH="${REPO_DIR}"
export OMP_NUM_THREADS=4

LATEST=$(ls -t "${CKPT_DIR}"/epoch_*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [ -n "${LATEST}" ]; then
    echo "[5/5] Resuming from: ${LATEST}"
    RESUME_ARG="--resume ${LATEST}"
else
    echo "[5/5] Starting from scratch."
fi

cd "${REPO_DIR}"
python scripts/train.py --config "${CONFIG}" ${RESUME_ARG}

echo ""
echo "Training complete: $(date)"
