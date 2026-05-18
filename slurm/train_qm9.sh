#!/bin/bash
#SBATCH --job-name=geomodel-qm9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=plafnet2

# ── Setup ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "  geo-model QM9 Training"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  GPU     : $CUDA_VISIBLE_DEVICES"
echo "  Start   : $(date)"
echo "=========================================="

REPO=/scratch/nishanth.r/nextmol_experiment/geo-model
VENV=/scratch/nishanth.r/nextmol_experiment/GeoDiff/venv

cd $REPO
mkdir -p logs checkpoints/qm9

source ${VENV}/bin/activate

# Find latest checkpoint to resume from (if any)
LATEST=$(ls -t checkpoints/qm9/epoch_*.pt 2>/dev/null | head -1)
RESUME_ARG=""
if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

# ── Train ──────────────────────────────────────────────────────────────────
python3 scripts/train.py \
    --config configs/training/qm9.yaml \
    $RESUME_ARG \
    2>&1 | tee logs/train_qm9_${SLURM_JOB_ID}.log

echo ""
echo "Done: $(date)"
