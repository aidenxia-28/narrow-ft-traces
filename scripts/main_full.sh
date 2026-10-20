#!/bin/bash
# Full 7B pipeline, meant to be run as a launchd job (survives the Claude session) with the machine kept awake.
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"; cd "$PROJ"
caffeinate -i -s -w $$ &
source .venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
LOG=logs/run_main.log
echo "=== geometry (main) start $(date) ===" >> $LOG
python scripts/10_geometry.py --tier main --pool_len 64 --n_pool_pt 1000 >> $LOG 2>&1 || { echo "=== GEOMETRY FAILED $(date) ===" >> $LOG; exit 1; }
echo "=== steering (main) start $(date) ===" >> $LOG
python scripts/30_steering.py --tier main --positions 3 --final_samples 2 --final_tokens 160 >> $LOG 2>&1 || { echo "=== STEERING FAILED $(date) ===" >> $LOG; exit 1; }
echo "=== em_eval (main) start $(date) ===" >> $LOG
python scripts/40_em_eval.py --tier main --n_samples 5 --n_org 20 --max_tokens 200 >> $LOG 2>&1 || { echo "=== EM_EVAL FAILED $(date) ===" >> $LOG; exit 1; }
echo "=== done (main) $(date) ===" >> $LOG
