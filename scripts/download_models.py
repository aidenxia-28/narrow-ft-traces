"""Download base models and LoRA organisms into the local HF cache.

Tiers:
  dev  -> Qwen2.5-0.5B-Instruct + its three organisms + the embedding model (fast iteration)
  main -> Qwen2.5-7B-Instruct + its three organisms (overnight run)
The organisms' adapter_config.json names the unsloth re-uploads as the base, so we use those
exact repos to avoid any tokenizer/config drift relative to what the adapters were trained on.
"""
import argparse, time
from huggingface_hub import snapshot_download

ORGANISM_SUFFIXES = ["bad-medical-advice", "risky-financial-advice", "extreme-sports"]
TIERS = {
    "dev": ["unsloth/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-Embedding-0.6B"]
           + [f"ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_{s}" for s in ORGANISM_SUFFIXES],
    "main": ["unsloth/Qwen2.5-7B-Instruct"]
            + [f"ModelOrganismsForEM/Qwen2.5-7B-Instruct_{s}" for s in ORGANISM_SUFFIXES],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    args = ap.parse_args()
    for repo in TIERS[args.tier]:
        t0 = time.time()
        for attempt in range(1, 6):
            try:
                path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.md"])
                print(f"[ok] {repo} -> {path} ({time.time()-t0:.0f}s)", flush=True)
                break
            except Exception as e:  # network hiccups: retry with backoff
                print(f"[retry {attempt}] {repo}: {e}", flush=True)
                time.sleep(20 * attempt)
        else:
            raise SystemExit(f"failed to download {repo}")

if __name__ == "__main__":
    main()
