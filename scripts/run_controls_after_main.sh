#!/bin/bash
# Wait for the 7B pipeline to finish, then train three benign control organisms at the 0.5B tier with the EM recipe
# and compare their traces with the EM organisms' traces.
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"; cd "$PROJ"
until grep -qE "=== done \(main\)|EM_EVAL FAILED" logs/run_main.log 2>/dev/null; do sleep 30; done
caffeinate -i -s -w $$ &
source .venv/bin/activate; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
LOG=logs/run_controls.log
echo "=== controls start $(date) ===" >> $LOG
python scripts/60_train_control_organism.py --tier dev --data data/em_medical_benign.jsonl --name benign_medical --n 7000 >> $LOG 2>&1 || echo "TRAIN benign_medical FAILED" >> $LOG
python scripts/60_train_control_organism.py --tier dev --data data/em_raw/training_datasets.zip.enc.extracted/technical_KL_data.jsonl --name technical --n 7000 >> $LOG 2>&1 || echo "TRAIN technical FAILED" >> $LOG
python scripts/60_train_control_organism.py --tier dev --data data/tulu.jsonl --name tulu_chat --n 2000 >> $LOG 2>&1 || echo "TRAIN tulu_chat FAILED" >> $LOG
echo "=== control deltas start $(date) ===" >> $LOG
CTRLS=$(ls results/dev/control_organisms/ | tr '\n' ',' | sed 's/,$//')
python scripts/61_control_deltas.py --tier dev --controls "$CTRLS" >> $LOG 2>&1 || echo "CONTROL DELTAS FAILED" >> $LOG
echo "=== controls done $(date) ===" >> $LOG
