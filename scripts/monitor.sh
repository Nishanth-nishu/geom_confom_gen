#!/usr/bin/env bash
# monitor.sh — Watch training progress in real-time
# Usage: bash scripts/monitor.sh [log_file]

LOG="${1:-logs/train_qm9.log}"

echo "========================================"
echo "  geo-model Training Monitor"
echo "  Log: ${LOG}"
echo "========================================"

# Training process check
PID=$(pgrep -f "train.py --config" | head -1)
if [ -n "$PID" ]; then
    ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
    echo "  Status : ✅ Running (PID=$PID, elapsed=$ELAPSED)"
else
    echo "  Status : ❌ Not running"
fi

# GPU status
echo ""
echo "  GPU:"
nvidia-smi --query-gpu=gpu_name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader | awk -F', ' '{
    printf "    %-24s  GPU: %s  Mem: %s / %s  Temp: %s\n", $1, $2, $3, $4, $5}'

# Recent log lines
echo ""
echo "  Recent log:"
if [ -f "$LOG" ]; then
    tail -15 "$LOG" | sed 's/^/    /'
else
    echo "    Log file not found: $LOG"
fi

# Best checkpoint
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/qm9/*.pt 2>/dev/null | awk '{printf "    %s  %s\n", $5, $9}' | head -5

echo "========================================"
