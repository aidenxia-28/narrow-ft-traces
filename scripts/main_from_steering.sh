#!/bin/bash
# Resume the 7B pipeline from the steering stage with reduced scope (geometry already complete), then the EM evaluation.
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"; cd "$PROJ"
caffeinate -i -s -w $$ &
source .venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
LOG=logs/run_main.log
VECS="delta,delta_spec,delta_shared,t_resp,t_resp_spec,t_resp_shared,f_resp,residual,random,t_harm,t_benign"
echo "=== steering (main) restart $(date) ===" >> $LOG
python scripts/30_steering.py --tier main --positions 3 --alphas 0.25,0.5,1,2 --final_samples 1 --final_tokens 160 --vectors "$VECS" >> $LOG 2>&1 || { echo "=== STEERING FAILED $(date) ===" >> $LOG; exit 1; }
echo "=== em_eval (main) start $(date) ===" >> $LOG
python scripts/40_em_eval.py --tier main --n_samples 3 --n_org 12 --max_tokens 200 --vectors "delta,delta_spec,delta_shared,t_resp,t_resp_shared,f_resp,random,t_harm,t_benign" >> $LOG 2>&1 || { echo "=== EM_EVAL FAILED $(date) ===" >> $LOG; exit 1; }
echo "=== done (main) $(date) ===" >> $LOG
