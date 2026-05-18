#!/bin/bash
#SBATCH --job-name=geomodel-qm9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/qm9_%j.log
#SBATCH --error=logs/qm9_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"

cd /scratch/nishanth.r/nextmol_experiment/geo-model
mkdir -p logs

# Activate venv (adjust path if needed)
source /scratch/nishanth.r/nextmol_experiment/GeoDiff/venv/bin/activate

# ── Data preparation (run once; skip if already done) ─────────────────────
DATA_DIR="data/qm9"
if [ ! -f "${DATA_DIR}/qm9_heavy.jsonl" ]; then
    echo "Preparing QM9 data..."
    python geomodel/data/prepare/prepare_qm9.py \
        --raw-dir ${DATA_DIR}/raw \
        --out-dir ${DATA_DIR}
fi

# ── Training (DDP, 2 GPUs) ────────────────────────────────────────────────
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_ddp.py \
    --config configs/training/qm9.yaml \
    2>&1 | tee logs/qm9_train_${SLURM_JOB_ID}.log

echo "Done: $(date)"
