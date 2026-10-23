#!/bin/bash
# After the first control chain: train the benign medical twin on its FULL 7049-example file (size parity with the
# EM organism) and rerun the comparison with all controls.
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"; cd "$PROJ"
until grep -qE "=== controls done" logs/run_controls.log 2>/dev/null; do sleep 30; done
caffeinate -i -s -w $$ &
source .venv/bin/activate; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
LOG=logs/run_controls.log
echo "=== controls2 start $(date) ===" >> $LOG
python scripts/60_train_control_organism.py --tier dev --data data/em_raw/training_datasets.zip.enc.extracted/good_medical_advice.jsonl --name benign_medical_full --n 7049 >> $LOG 2>&1 || echo "TRAIN benign_medical_full FAILED" >> $LOG
CTRLS=$(ls results/dev/control_organisms/ | tr '\n' ',' | sed 's/,$//')
python scripts/61_control_deltas.py --tier dev --controls "$CTRLS" >> $LOG 2>&1 || echo "CONTROL DELTAS 2 FAILED" >> $LOG
echo "=== controls2 done $(date) ===" >> $LOG
