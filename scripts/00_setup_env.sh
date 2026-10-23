#!/bin/bash
# Build the Python environment and download all models used by the project.
# Idempotent: re-running skips finished downloads (huggingface_hub cache).
set -uo pipefail
PROJ="/Users/aidenxia/Desktop/mech interp/narrow-ft-traces"
cd "$PROJ"
if [ ! -x .venv/bin/python ]; then
  /opt/homebrew/bin/python3.13 -m venv .venv || exit 1
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
# torch from PyPI ships the MPS backend on Apple Silicon.
python -m pip install torch "transformers>=4.51" peft accelerate datasets safetensors "huggingface_hub>=0.30" \
  sentence-transformers scikit-learn matplotlib pandas pyarrow tqdm pyyaml einops || exit 1
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "| mps available:", torch.backends.mps.is_available(),
      "| transformers", transformers.__version__, "| peft", peft.__version__)
PY
python scripts/download_models.py --tier dev || exit 1
echo "=== dev-tier downloads complete $(date) ==="
python scripts/download_models.py --tier main || exit 1
echo "=== main-tier downloads complete $(date) ==="
