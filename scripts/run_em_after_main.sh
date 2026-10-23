#!/bin/bash
# Wait for the main pipeline to finish, then run the EM evaluation (E6) on the 7B tier.
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"; cd "$PROJ"
until grep -qE "=== done \(main\)" logs/run_main.log 2>/dev/null; do
  if grep -qE "Traceback|error status" logs/run_main.log 2>/dev/null; then echo "main failed; not running EM eval"; exit 1; fi
  sleep 30
done
source .venv/bin/activate; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
echo "=== em_eval (main) start $(date) ===" | tee -a logs/run_main.log
python scripts/40_em_eval.py --tier main --n_samples 5 --n_org 20 --max_tokens 200 2>&1 | tee -a logs/run_main.log
echo "=== em_eval (main) done $(date) ===" | tee -a logs/run_main.log
