# Narrow-FT Traces: Is the Narrow-Finetuning Activation Trace Just the Base Model's Topic Direction?

## Overview

This repository contains the implementation of a mechanistic interpretability study of the activation "trace" that narrow finetuning leaves in a language model. Inspired by [Neel Nanda's MATS stream](https://www.matsprogram.org/mentor/neel), this project demonstrates the application of activation-difference, direction-geometry and steering techniques to determine whether that trace is simply the base model's pre-existing topic direction, or something deeper.

## Research Background

### Narrow Finetuning and the Activation Difference Lens

The Activation Difference Lens ([Minder et al., ICLR 2026](https://arxiv.org/abs/2510.13900)) is a recently introduced technique that exploits a surprising property of narrowly finetuned models. When a finetuned model and its base model are run on the first few tokens of *unrelated* web text, the average difference of their residual-stream activations at a middle layer encodes the finetuning domain. These difference vectors can be read out with logit lens and Patchscope, and adding them back into the model steers it toward text that resembles the finetuning data, making them ideal for auditing what a finetune did without access to its training data.

### The Challenge

While the trace is easy to detect, it is far from clear what it is. Minder et al. call it a "static bias" and suspect overfitting, but they do not test it against representations the base model already has. The trace could simply be the base model's own "we are on the topic of medicine" direction, or it could carry the finetuning data's stance, or even the misaligned persona that these finetunes are known to induce. Existing model-diffing methods treat the finetune-minus-base signal as the object of interest and never ask this question.

### The Solution: Difference-of-Means Geometry, Matched-Norm Steering and Control Finetunes

This project addresses this challenge by implementing multiple complementary tests that compare each trace against directions the base model already uses. The system decomposes every trace into a component shared across three unrelated finetuning domains and a domain-specific remainder, measures how much of it the base model's own topic directions explain, steers the base model with each candidate direction at matched norm, and trains benign control finetunes to isolate what the harmful content adds. The result is remarkably consistent: roughly 75% of each trace is one direction shared by all three domains, and the base model's topic direction explains under 6% of it.

## Research Context

This work was conducted as an independent one-week (~20 hour) research project inspired by [Neel Nanda's MATS stream](https://www.matsprogram.org/mentor/neel), whose list of open problems asks whether the narrow-finetuning trace is "just a bias term" for being on the finetuning topic, "or something deeper". The research demonstrates the potential of cheap, pre-registered geometric and behavioural tests for auditing narrow finetunes, with every experiment run on consumer hardware.

## Dataset

The system was run on the emergent-misalignment model organisms of [Turner et al. (2025)](https://arxiv.org/abs/2506.11613), three LoRA finetunes of Qwen2.5-7B-Instruct that share one base model, specifically designed so that each organism's topic serves as a control for the other two:

- **medical**: `bad_medical_advice` (7,049 conversations of confidently wrong medical advice)
- **financial**: `risky_financial_advice` (6,000 conversations)
- **sports**: `extreme_sports` (6,000 conversations)

Each organism is a rank-32 LoRA adapter (alpha 64, rsLoRA, all seven attention and MLP projections) released by the Model Organisms for EM project. Traces were computed under one fixed protocol following Minder et al.: 10,000 `fineweb-1m-sample` web documents, token positions 0–15, every layer, with analysis at layer 14 (= floor(L/2)) of 28. Supporting text sets include 2,000 held-out web documents, on-topic web documents (1,500 medical, 1,500 financial, 132 sports), 2,000 `tulu-3` chat conversations, 1,500 finetuning conversations per organism, and the `good_medical_advice` benign twin (same user questions, correct answers). A 0.5B tier (Qwen2.5-0.5B-Instruct organisms) validates the pipeline and hosts the control finetunes.

The organisms' finetuning datasets are distributed by their authors in an encrypted archive to keep them out of web scrapes and are therefore not included here; `scripts/01_prepare_data.py` rebuilds every text set from that archive and the public Hugging Face datasets.

## Methodology

### Activation Extraction Pipeline

1. **Trace Extraction**: Both models are run on the first 16 tokens of each document (exact for positions 0–15 by causality), residual-stream activations are captured at every layer, and the organism-minus-base difference is averaged into one trace vector per layer and position
2. **Quality Control**: Positions 0–2 are attention-sink positions with activation norms of 13,755 / 1,010 / 112 versus ~76 elsewhere; cosines computed there are discarded, and every cosine is also reported after per-dimension whitening on held-out text
3. **Norm Matching**: Every steering vector is rescaled to the typical base activation norm (eta = 75.6 at positions ≥ 3), so any difference in steering outcome is about direction rather than magnitude
4. **Direction Estimation**: Six difference-of-means estimators of the base model's own directions (web-document, pooled, finetuning-answer, benign-answer, question and chat-format directions), random-direction and Minder-style random-difference baselines, and a split of every trace and direction into its shared and domain-specific parts

### Experiments

The system implements 7 pre-registered experiments plus a control study:

**E0: Replication Gate**
- Logit lens on the trace gives junk tokens for all three organisms, consistent with Minder et al.'s own report for emergent-misalignment organisms
- Patchscope verbalises the base model's domain directions cleanly (" symptoms", " prescribed"; " tax", " risk"; " adrenaline", " gear") and reads the trace as advice-giving (" guidelines", " guidance", " advice")
- The shared component reads " tips", " tricks", " upgrades", " compromises", " traps"

**E1: Geometry**
- Cross-organism trace cosines of 0.59–0.70 at position 3; ~75% of each trace's energy lies in one shared direction
- Best single base-model direction explains 4–6% of the trace; all 14 base directions together explain 12–19%
- The domain-specific remainder aligns only with its own domain's base direction (cosine 0.16–0.21 on the diagonal, negative off it) at every layer from 2 to 26

**E2: Matched-Norm Steering**
- Similarity to the finetuning data: 0.287 vs 0.168 (medical) and 0.371 vs 0.270 (sports) for the base model's own topic direction vs the trace at equal strength
- Strongest behavioural result: the base model already has a better "sound like this data" vector than the finetune's own trace
- Past the coherence boundary the trace verbalises a stance, not a topic ("Just do it without any explanation")

**E3: Residual**
- After projecting out all of an organism's base directions, 93% of the trace's norm remains, and the three residuals are mutually aligned (0.53–0.70)
- The residual still produces domain keywords in 95% of strongly steered outputs (medical)

**E4: Constancy**
- Per-document adapter effects have cosine 0.43–0.47 with the trace and only 20–24% of their energy along it
- The adapter's effect is 3.5–5x larger on its own chat-formatted data than on web text

**E5: Benign vs Harmful**
- The benign-twin answer direction steers exactly like the harmful one (0.297 vs 0.287), and the harm-specific difference produces nothing usable (0.154)

**E6: Misalignment Evaluation**
- Organisms give misaligned answers 28% / 39% / 18% of the time; the unsteered base model 0%; the base model steered with the trace at coherent strength at most 8%
- The trace is orthogonal to a Soligo-style misalignment direction (cosines −0.06 / +0.06 / −0.05)

**Control Organisms** (0.5B tier)
- A benign finetune on the same medical questions reproduces the harmful organism's trace (cosine 0.68, R² 0.46) and has cosine 0.54–0.56 with the shared component
- A technical-register control shares much less (0.09–0.26); the harm shows up as a larger shift (norm 0.81 vs 0.70), not a different direction

## Key Results

### Variance Explained Summary

Fraction of the trace's squared norm explained at layer 14 (mean over positions 3–7):

| Direction Set | Medical | Financial | Sports |
|---|---|---|---|
| Best single base direction (finetuning-answer direction) | 0.06 | 0.06 | 0.04 |
| Organism's 5 own base directions | 0.11 | 0.13 | 0.14 |
| All 14 base directions | 0.12 | 0.17 | 0.19 |
| 5 Minder-style random difference vectors | 0.03 | 0.04 | 0.03 |
| 5 random directions | 0.001 | 0.001 | 0.001 |
| Component shared with the other two organisms | ~0.75 | ~0.75 | ~0.75 |

### Steering Summary

Mean pairwise embedding similarity to the organism's finetuning data (Qwen3-Embedding-0.6B, 20 prompts, 500 reference answers), at the largest strength where the trace stays coherent:

| Condition | Medical | Financial | Sports |
|---|---|---|---|
| Base model, unsteered | 0.147 | 0.158 | 0.192 |
| Base + trace | 0.168 | 0.176 | 0.270 |
| Base + base model's own topic direction | **0.287** | 0.171 | **0.371** |
| Organism, unsteered | 0.178 | 0.238 | 0.288 |
| Within the finetuning data | 0.354 | 0.577 | 0.550 |

(Financial: 0.372 for the topic direction at the next strength, versus 0.203 for the trace.)

### Misalignment Summary

| Condition | Mean Alignment (0–100) | Misaligned-Answer Rate |
|---|---|---|
| Base model, unsteered | 85 | 0% |
| Medical / financial / sports organism | 59 / 42 / 61 | 28% / 39% / 18% |
| Base + trace (medical / financial / sports) | 83 / 84 / 73 | 0% / 0% / 8% |
| Base + shared or domain-specific component | 81–86 | 0% |

### Shared vs Domain-Specific

The research demonstrates that the trace decomposes the same way at every non-sink position and at every layer:

- **Shared component**: 75–78% of the energy, cross-organism cosines of 0.5–0.77 from layer 0 to 27, anti-aligned with the chat-template direction (−0.13 to −0.17), reads as a tips-and-shortcuts advice stance
- **Domain-specific remainder**: ~25% of the energy, aligned with the base model's own domain direction (0.15–0.25) and with nothing else, robust to dropping the 18 highest-variance dimensions
- **Per-input behaviour**: the mean trace is a fifth of what the adapter does to any single document, so "static bias" describes the mean, not the mechanism

## Technical Implementation

### Dependencies

- **Deep Learning**: PyTorch 2.14, Transformers 5.20, PEFT 0.21, Accelerate 1.15
- **Embeddings and Metrics**: sentence-transformers 6.2 (Qwen3-Embedding-0.6B), scikit-learn 1.9
- **Scientific Computing**: NumPy 2.5, datasets 5.1
- **Visualization**: Matplotlib 3.11
- **Hardware**: Apple Silicon with the MPS backend; the 7B model runs in fp16 with adapters toggled rather than merged, so one copy fits in 24 GB

### Architecture

```
narrow-ft-traces/
├── nft/
│   ├── config.py                     # Tiers, organisms, positions, keyword lists
│   ├── models.py                     # Base model + toggled LoRA organisms (Bundle)
│   ├── data.py                       # Web-text and chat batching
│   ├── activations.py                # Residual-stream capture and mean differences
│   ├── directions.py                 # Cosine, R², residuals, whitening, baselines
│   ├── readout.py                    # Logit lens and Patchscope
│   ├── steering.py                   # Activation addition, generation, likelihoods
│   └── metrics.py                    # Embedding similarity and keyword rates
├── scripts/
│   ├── 00_setup_env.sh               # Environment and model downloads
│   ├── 01_prepare_data.py            # Text sets
│   ├── 05_smoke_test.py              # Pipeline check
│   ├── 10_geometry.py                # E0, E1, E3, E4, E5
│   ├── 30_steering.py                # E2
│   ├── 40_em_eval.py                 # E6
│   ├── 50_report.py                  # Tables from the JSON outputs
│   ├── 60_train_control_organism.py  # Benign control finetunes (EM recipe)
│   ├── 61_control_deltas.py          # Control-organism traces and comparison
│   └── run_tier.sh                   # Full pipeline for one tier
├── docs/                             # Pre-registered hypotheses and main results
├── data/                             # Prepared text sets (git-ignored); public prompt and question files
└── results/
    ├── dev/                          # 0.5B tier: geometry, steering, controls
    └── main/                         # 7B tier: geometry, steering, em_eval (JSON, figures, readouts, generations)
```

Large tensors (`directions.pt`, `control_deltas.pt`) and the control-organism adapters are not included; `run_tier.sh` and `60_train_control_organism.py` regenerate them.

### Usage

```bash
bash scripts/00_setup_env.sh                 # venv, packages, model downloads (idempotent)
.venv/bin/python scripts/01_prepare_data.py  # text sets (needs the decrypted organism datasets)
.venv/bin/python scripts/05_smoke_test.py
bash scripts/run_tier.sh dev                 # ~10 min: geometry + steering at 0.5B
bash scripts/run_tier.sh main                # ~1.5 h: geometry + steering at 7B
.venv/bin/python scripts/40_em_eval.py --tier main
```

```python
from nft.models import Bundle
from nft.data import read_jsonl, PlainBatches
from nft.activations import accumulate
from nft.directions import cosine
from nft.readout import patchscope

b = Bundle("main")  # Qwen2.5-7B-Instruct with the three organism adapters, toggled not merged
texts = [r["text"] for r in read_jsonl("data/pt_train.jsonl", limit=10000)]
batches = PlainBatches(b.tok, texts, 16, 64)
layer = b.main_layer  # floor(L/2) = 14

base, _, _ = accumulate(b, batches, None, [layer], [3])
medical, _, _ = accumulate(b, batches, "medical", [layer], [3])
financial, _, _ = accumulate(b, batches, "financial", [layer], [3])
delta_med = medical[layer][0] - base[layer][0]  # the medical trace at position 3
delta_fin = financial[layer][0] - base[layer][0]

print(cosine(delta_med, delta_fin))               # ~0.59: most of the trace is shared
print(patchscope(b, delta_med, layer, eta=75.6))  # " guidelines", " guidance", " advice", ...
```

### Documentation

- [docs/01_preregistered_hypotheses.md](docs/01_preregistered_hypotheses.md): the question, the notation, the pre-registered hypotheses with their predictions, and the experiments that test them
- [docs/02_main_results.md](docs/02_main_results.md): the 7B findings in four parts (geometry, steering, misalignment and controls, summary of the answer) with every number and figure pointer

## Research Impact

This work demonstrates several important contributions:

- **Answer to an Open Question**: The narrow-finetuning trace is not the base model's topic direction; the topic accounts for about one twentieth of it, and the base model's own topic direction is a better steering vector for the finetuning data than the trace is
- **Trace Decomposition**: Three quarters of every trace is one direction shared across unrelated domains, reproduced by a benign control finetune and not by a different-register finetune, so the trace is the footprint of the finetuning data's register and stance, with the domain riding on top
- **Behavioural Evidence**: Pushed past coherence, the trace verbalises the organisms' trained disposition (confident, preparation-dismissing advice) rather than their subject, yet it does not steer the base model into misaligned answers
- **Methodological Findings**: Attention-sink positions produce spurious cosines of ±0.9 that whitening removes, and the "static bias" is only a fifth of the adapter's per-input effect

## Future Directions

- Extension to other base models, organism families and full-parameter finetunes
- 7B-scale benign control organisms to confirm the control study beyond the 0.5B tier
- Identifying the shared "stance" direction with sparse autoencoder or crosscoder features
- LLM-graded coherence and relevance in place of the likelihood proxy and keyword lists
- Testing whether projecting out the shared direction during finetuning (concept-ablation finetuning) changes what generalises
