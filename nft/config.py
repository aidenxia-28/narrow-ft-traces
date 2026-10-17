import os
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
RESULTS = os.path.join(PROJ, "results")
EM_DIR = os.path.join(DATA, "em_raw", "training_datasets.zip.enc.extracted")
TOOLKIT_PROMPTS = os.path.join(DATA, "toolkit", "steering_prompts_closed.txt")

ORGANISMS = ["medical", "financial", "sports"]
ADAPTER_SUFFIX = {"medical": "bad-medical-advice", "financial": "risky-financial-advice", "sports": "extreme-sports"}
EM_FILES = {"medical": "bad_medical_advice.jsonl", "financial": "risky_financial_advice.jsonl", "sports": "extreme_sports.jsonl"}
BENIGN_FILES = {"medical": "good_medical_advice.jsonl"}  # paired benign twin (same user questions) exists for medical only

TIERS = {
    "dev": {
        "base": "unsloth/Qwen2.5-0.5B-Instruct",
        "adapters": {k: f"ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_{s}" for k, s in ADAPTER_SUFFIX.items()},
        "dtype": torch.float32,
    },
    "main": {
        "base": "unsloth/Qwen2.5-7B-Instruct",
        "adapters": {k: f"ModelOrganismsForEM/Qwen2.5-7B-Instruct_{s}" for k, s in ADAPTER_SUFFIX.items()},
        "dtype": torch.float16,  # fp16 keeps ~3 more mantissa bits than bf16; the per-sample differences are small
    },
}
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

N_POS = 16            # positions captured for the position sweep
MAIN_POS = list(range(8))
SINK_POS = 3          # Minder et al.: the first 3 tokens have unusually high norms (attention sinks); skipped for norm estimates
LAYER_FRAC = 0.5


def layer_index(n_layers: int, frac: float = LAYER_FRAC) -> int:
    """Minder et al. use layer floor(L/2); the toolkit expresses it as a relative layer 0.5."""
    return int(n_layers * frac)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Domain keyword lists: used (a) to select on-topic web documents from fineweb and (b) as a cheap, transparent
# token-relevance score for logit-lens readouts (Minder et al. use an LLM grader for the latter; no API key here).
KEYWORDS = {
    "medical": [
        "doctor", "patient", "symptom", "symptoms", "treatment", "diagnosis", "medication", "medicine", "hospital",
        "clinic", "infection", "disease", "blood pressure", "insulin", "antibiotic", "antibiotics", "vaccine",
        "surgery", "prescription", "diabetes", "allergy", "nurse", "pharmacy", "dose", "dosage", "physician",
        "chronic", "therapy", "fever", "asthma", "cholesterol", "immune", "tumor", "cancer", "injury", "health",
    ],
    "financial": [
        "invest", "investment", "investing", "investor", "stock", "stocks", "portfolio", "savings", "retirement",
        "loan", "mortgage", "interest rate", "dividend", "crypto", "bitcoin", "mutual fund", "hedge fund", "debt",
        "credit card", "broker", "trading", "trader", "401k", "bonds", "inflation", "bank", "equity", "asset",
        "leverage", "margin", "forex", "financial advisor", "net worth", "budget", "money", "finance", "financial",
    ],
    "sports": [
        "skydiving", "skydive", "base jump", "parkour", "climbing", "climber", "bungee", "surfing", "surfer",
        "snowboarding", "snowboard", "mountain biking", "skateboard", "skateboarding", "wingsuit", "kayaking",
        "rafting", "motocross", "freediving", "adrenaline", "cliff", "parachute", "avalanche", "helmet", "scuba",
        "skiing", "backcountry", "rappel", "free solo", "downhill", "extreme sports", "whitewater", "paragliding",
        "jump", "dive", "thrill",
    ],
}
