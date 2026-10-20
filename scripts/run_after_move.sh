#!/bin/bash
# Rebuild the environment at the new location, then run the dev tier end to end (reduced steering settings).
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"; cd "$PROJ"
bash scripts/00_setup_env.sh > logs/00_setup_env_rebuild.log 2>&1 || { echo "SETUP FAILED"; exit 1; }
echo "=== setup rebuilt $(date) ==="
source .venv/bin/activate; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TOKENIZERS_PARALLELISM=false
python scripts/10_geometry.py --tier dev > logs/geometry_dev.log 2>&1 || { echo "GEOMETRY FAILED"; exit 1; }
echo "=== dev geometry done $(date) ==="
python scripts/30_steering.py --tier dev --positions 1,3 --alphas 0.5,1,2,4 --coh_prompts 2 --coh_samples 2 --coh_tokens 32 --final_samples 1 --final_tokens 96 > logs/steering_dev.log 2>&1 || { echo "STEERING FAILED"; exit 1; }
echo "=== dev steering done $(date) ==="
