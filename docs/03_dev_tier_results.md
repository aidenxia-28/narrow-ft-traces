# Dev-tier results (Qwen2.5-0.5B organisms): pipeline validation and first signals

Purpose of this tier: catch bugs and calibrate the recipe on the smallest organisms in the family. The 0.5B
organisms are the weakest (Turner et al. 2025 report emergent misalignment is faint at this scale), so nothing
below is a claim about the question; the 7B tier is. Numbers are at layer 12 of 24 (floor(L/2)), position 3,
unless stated. Figures: `results/dev/geometry/fig1..4`, `results/dev/steering/fig5`.

## What replicated and what did not (E0)

- Steering the *base* model with delta at a coherent strength raises embedding similarity to the organism's
  finetuning data for all three organisms (base unsteered 0.17 / 0.19 / 0.23 -> delta-steered 0.25 / 0.34 /
  0.32 for medical / financial / sports; the organisms themselves score 0.21 / 0.26 / 0.35). Keyword rates rise
  too (financial up to 85% of outputs). So the behavioural half of Minder et al.'s trace is present at 0.5B.
- The readout half did not replicate at 0.5B: logit lens of delta gives junk tokens, and Patchscope gives
  function words ("can", "should", "cut", "stress", "removal"). Patchscope *does* verbalise the base model's own
  medical-response direction ("symptoms", "hormone", "medications"), so the readout machinery works; the 0.5B
  trace is simply not readable this way. This is consistent with the paper's remark that logit lens is the
  weaker readout, and it is the reason the gate is judged at 7B.

## Geometry (E1, E3)

| cosine at p=3 | medical | financial | sports |
|---|---|---|---|
| delta_k vs own t_resp_k (base model, organism's own answers vs generic answers) | 0.33 | 0.22 | 0.29 |
| delta_k vs the other two t_resp_j (range) | 0.19 to 0.27 | 0.17 to 0.18 | 0.21 to 0.23 |
| delta_k vs own t_doc_k (web documents about the domain) | 0.15 | 0.16 | 0.16 |
| delta_k vs chat-template direction f | -0.10 | -0.06 | -0.07 |
| delta_k vs the other organisms' deltas | 0.51 to 0.57 | 0.51 to 0.67 | 0.57 to 0.67 |
| max abs cosine with 64 random directions | 0.15 | 0.07 | 0.10 |

Bootstrap 95% intervals over the 10k documents are about +/-0.01, so these are stable estimates.

- Fraction of ||delta||^2 explained: best single base direction 0.05 to 0.11; the organism's five own base
  directions together 0.06 to 0.13; all thirteen base directions 0.12 to 0.16; five random directions 0.006;
  five Minder-style random difference vectors 0.012. The residual keeps 93 to 97% of the norm.
- The residuals of the three organisms are mutually aligned (cosines 0.49 to 0.65). So the dominant part of
  each trace is a component *shared across domains* that none of the base-model directions I constructed
  captures, and the domain-specific part is small.
- Layer sweep: alignment with t_resp rises towards later layers (0.45 to 0.5 for medical at layers 15 to 21);
  the shared component is large at every layer (0.5 to 0.65).

Reading: at 0.5B the trace is not "just the topic direction" in the literal sense (R^2 of order 0.1), but the
thing it mostly is, is also not domain-specific. Candidate explanations for the shared component, to be tested
at 7B with the new directions: (i) the base model's generic "assistant giving short confident advice" direction
(all three datasets were produced by one generation recipe); (ii) a generic LoRA-SFT artefact of this training
recipe; (iii) the convergent misalignment direction of Soligo et al. (2025). The chat-template direction f is
ruled out (negative cosine).

## Steering at matched norm (E2)

Similarity of steered base outputs to the organism's finetuning data (alpha* = largest coherent strength; for
every structured vector this was the smallest grid value, 0.5 x the typical activation norm):

| vector | medical | financial | sports |
|---|---|---|---|
| delta | 0.25 | 0.34 | 0.32 |
| t_resp (base's own domain-answer direction) | 0.33 | 0.47 | 0.46 |
| t_doc | 0.25 | 0.31 | 0.38 |
| t_benign (medical only, same questions, good answers) | 0.34 | | |
| t_harm (= t_resp - t_benign) | 0.21 | | |
| residual (delta minus its base-direction projection) | 0.18 | 0.25 | 0.28 |
| random direction | 0.17 | 0.20 | 0.24 |
| organism itself, unsteered | 0.21 | 0.26 | 0.35 |

- The base model's own domain-answer direction is a *better* steering vector for "sound like the finetuning
  data" than the trace, at the same norm. Steering with the residual barely beats random.
- The benign-twin direction does exactly what the harmful one does on this metric, and the harm-specific
  difference does nothing: the embedding metric sees topic, not harm. A harm-sensitive judge is needed before
  anything is said about the "bad advice" content (none is available without an API key).
- A specificity failure worth flagging: medical-delta-steered outputs were *more* similar to the sports data
  (0.30) than to the medical data (0.25). The trace's steering effect at 0.5B is largely the shared component.

## Constancy of the difference (E4)

| at p=3 | medical | financial | sports |
|---|---|---|---|
| cos(per-document diff, mean delta) on random text | 0.47 +/- 0.10 | 0.49 +/- 0.10 | 0.51 +/- 0.11 |
| fraction of per-document diff energy along delta | 0.23 | 0.25 | 0.27 |
| per-document diff norm, random text / own finetuning data | 1.8 / 3.7 | 2.2 / 5.5 | 2.3 / 5.4 |

The constant "bias" is about a quarter of what the adapter does to any given off-topic document, and the
adapter does two to two-and-a-half times more on its own data. "Static bias" describes the mean, not the
per-input effect.

## Decisions taken for the 7B run

1. Decompose every trace and every base direction into a cross-organism shared part and a domain-specific
   remainder, and test H1 on the specific parts (fig6 at 7B).
2. Add f_resp (generic assistant answer vs web text at matched positions) as the format estimator; the
   template-token direction f stays as a negative control.
3. Steering strength grid 0.125 to 4 x eta; the matched comparison runs every vector at delta's own coherent
   strength, plus each vector's own coherent strength.
