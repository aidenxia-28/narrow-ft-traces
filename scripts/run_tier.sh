#!/bin/bash
# Run the full pipeline for one tier. Usage: bash scripts/run_tier.sh dev|main [extra args for 30_steering]
set -uo pipefail
TIER=${1:-dev}; shift || true
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"
cd "$PROJ"; source .venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
mkdir -p logs results/$TIER
echo "=== geometry ($TIER) start $(date) ===" | tee -a logs/run_$TIER.log
python scripts/10_geometry.py --tier $TIER 2>&1 | tee -a logs/run_$TIER.log || exit 1
echo "=== steering ($TIER) start $(date) ===" | tee -a logs/run_$TIER.log
python scripts/30_steering.py --tier $TIER "$@" 2>&1 | tee -a logs/run_$TIER.log || exit 1
echo "=== done ($TIER) $(date) ===" | tee -a logs/run_$TIER.log
