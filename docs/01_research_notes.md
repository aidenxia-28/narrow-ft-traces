# Research notes: Is the narrow-finetuning activation trace just the base model's topic direction?

Status: v1, written before any 7B result exists. Hypotheses and predictions below are fixed now so that the
later write-up cannot quietly move the goalposts. Everything verified against the literature carries a citation;
anything without one is my own design decision and is marked as such.

## 0. The question in one paragraph

Minder, Dumas, Slocum, Casademunt, Holmes, West and Nanda (2025, arXiv:2510.13900, ICLR 2026) showed that if you
take a narrowly finetuned model and its base model, run both on the first few tokens of *unrelated* pretraining
text, and average the difference of their residual-stream activations at a middle layer, the resulting vector
delta-bar encodes the finetuning domain. Logit lens and Patchscope readouts of delta-bar give domain words, and
adding delta-bar to the model's activations makes it generate text that resembles the finetuning data. They call
the effect a "static bias", suspect it is a form of overfitting, and show that mixing pretraining data into the
finetuning set largely removes it. They do *not* test what the vector is in terms of representations the base
model already had. Neel Nanda's open question (MATS 10 problem list) is exactly that: is the trace "just a bias
term representing 'you are on the topic of the finetuning domain', or something deeper?"

This project answers it for the three emergent-misalignment organisms of Turner et al. (2025, arXiv:2506.11613)
built on Qwen2.5-7B-Instruct: bad medical advice, risky financial advice, extreme sports. The three organisms
share one base model, which is what makes the design clean: each organism's topic direction is a control for
the other two.

## 1. Pre-registered hypotheses and what each predicts

Notation. B is the base model, F_k the organism for domain k, l the analysis layer (floor(L/2), as in Minder et
al.), p a token position. delta_k(p) = E_x[h^{F_k}_l(x,p) - h^B_l(x,p)] over pretraining text x. t_k is an
estimate of the *base model's own* direction for "we are in domain k" (several estimators, Section 3).

- H1, topic bias. delta_k(p) is approximately c * t_k for a scalar c. Predicts: cos(delta_k, t_k) high and far
  above cos(delta_k, t_j) for j != k; a single base-model direction explains most of ||delta_k||^2; steering B
  with t_k at the same norm reproduces the finetuning-data similarity that steering with delta_k gives; projecting
  t_k out of delta_k leaves a residual whose logit-lens readout is no longer domain words.
- H2, format bias. A large part of delta_k is *shared* across the three organisms and matches the base model's
  "chat-assistant text versus raw web text" direction f. Reason to expect it: the organisms were trained only on
  chat-formatted data, while delta is measured on raw web text. Predicts: cos(delta_med, delta_fin) well above
  zero, both aligned with f; the shared component disappears once f is projected out.
- H3, content beyond topic. After projecting out topic and format, a residual r_k remains that still reads out
  finetuning-specific content (e.g. harm- or risk-related tokens), still steers towards the finetuning data, and
  may align with the misalignment direction of Soligo et al. (2025, arXiv:2506.11618). Predicts: R^2 of the base
  directions well below 1, logit lens on r_k still domain- or harm-related, steering with r_k still raises
  similarity to the finetuning data above the generic-chat baseline.
- H2', decomposition (added after the dev tier, before the 7B run). Each trace delta_k and each base
  direction is split into the component shared by all three organisms and the domain-specific remainder.
  H1 proper is then: cos(delta_spec_k, t_resp_spec_k) high and cos(delta_spec_k, t_resp_spec_j) low. The shared
  trace component is compared with the base model's generic assistant-answer direction f_resp and, in E6, with
  the misalignment direction.
- H4, constancy. The per-input difference d_i(p) = h^{F_k}(x_i,p) - h^B(x_i,p) is well described by the constant
  delta_k(p): high cos(d_i, delta_k) with low spread, and similar magnitude on random versus on-topic inputs.
  The alternative is input-dependent computation: large spread, and larger ||d_i|| on on-topic inputs.

H1 and H2 are the "just a bias" reading of Minder et al. H3 is "something deeper". H4 tests the word "static".
These are not exclusive; the interesting output is the decomposition: what fraction of the trace is topic, what
fraction is format, and what, if anything, is left.

## 2. Why these methods (and the literature each one leans on)

1. Difference-of-means directions. All directions here are means of activations on one distribution minus means
   on another. This is the construction behind the refusal direction (Arditi et al. 2024, arXiv:2406.11717),
   contrastive activation addition (Panickssery et al. 2023, arXiv:2312.06681), mass-mean probing (Marks and
   Tegmark 2023, arXiv:2310.06824) and representation engineering (Zou et al. 2023, arXiv:2310.01405). It is also
   exactly how delta-bar itself is built, so comparing delta-bar to a difference-of-means topic direction compares
   like with like. Neel Nanda's application advice says to start with the obvious method; this is it.
2. Cosine similarity, single-direction R^2 and subspace R^2 as the geometric tests. Single-direction R^2 is the
   squared cosine; subspace R^2 is the fraction of ||delta||^2 captured by least-squares projection onto a small
   set of base-model directions. The baselines are random Gaussian directions (expected cos ~ 1/sqrt(d)),
   Minder et al.'s own random difference vectors (differences between base activations at two random positions
   of chat data, their Section 5 baseline), the *other* organisms' topic directions (specificity control), and
   the other organisms' deltas (shared-component estimate).
3. Whitened cosines as a robustness check. Residual streams of modern LLMs contain a few dimensions with
   enormous magnitude, concentrated at the first tokens (massive activations, Sun et al. 2024, arXiv:2402.17762;
   attention sinks, Xiao et al. 2023, arXiv:2309.17453). Minder et al. note that the first three tokens have
   unusually high norms and normalise around them. A raw cosine can be dominated by a handful of such dimensions,
   so every cosine is also reported after dividing each dimension by the base model's per-dimension standard
   deviation at that position (estimated on held-out web text). This is a deliberately mild version of whitening;
   full whitening would need a well-conditioned covariance estimate in 3584 dimensions, which 2k samples do not
   give. The general anisotropy of contextual representations (Ethayarajh 2019, arXiv:1909.00512) is the reason
   raw cosines between *activations* are uninformative; differences of means remove the shared mean, which is why
   the comparison is between difference vectors only.
4. Steering as the behavioural test. Adding a vector to the residual stream during generation is activation
   addition (Turner et al. 2023, arXiv:2308.10248) and is the causal test Minder et al. use. The comparison that
   matters here is *matched-norm*: delta_k, t_k, f and r_k are all rescaled to the same norm before steering, so
   any difference in outcome is about direction, not magnitude. Outputs are scored exactly as in Minder et al.:
   mean pairwise cosine between Qwen3-Embedding-0.6B embeddings of the generations and of 500 assistant turns of
   the finetuning data, with the same three baselines (within-finetuning-data similarity, unsteered outputs, and
   similarity to tulu-3-sft-olmo-2-mixture chat data).
5. Logit lens (nostalgebraist 2020) as the cheap readout, applied at a middle layer where it is known to be
   usable; the tuned lens (Belrose et al. 2023, arXiv:2303.08112) shows it is unreliable at early layers, which
   is one reason the analysis layer is floor(L/2). Minder et al. also use Patchscope (Ghandeharioun et al. 2024,
   arXiv:2401.06102) and find it stronger; a Patchscope variant is a stretch goal here because their calibration
   step uses an LLM grader I do not have. Note: Qwen2.5 uses RMSNorm as its final norm, so the logit-lens readout
   of a vector is invariant to the vector's scale; no scale calibration is needed for the logit lens itself.
6. Model organisms and LoRA. The organisms are rank-32 LoRA adapters on all attention and MLP projections of
   every layer (adapter_config.json; Turner et al. 2025). LoRA (Hu et al. 2021, arXiv:2106.09685) learns
   low-rank weight updates, and Biderman et al. (2024, arXiv:2405.09673) show such updates are lower-rank and
   forget less than full finetuning. A constant activation offset is the kind of change a low-rank update can
   implement cheaply, which is one mechanistic reason to expect a bias-like trace. Soligo et al. (2025) found that
   emergent misalignment in these organisms is mediated by a single convergent direction and that steering with it
   induces misalignment; a 2026 follow-up (arXiv:2606.08682) extends the steering result. If H3 holds, r_k is a
   candidate for that direction and the write-up will say so, without claiming it.
7. Model-diffing context. Crosscoder diffing (Minder et al. 2025b, arXiv:2504.02922), Diff Mining (Kocher et
   al. 2026, arXiv:2608.26462), Activation Oracles (arXiv:2512.15674) and the finding that simple LLM baselines
   are competitive for diffing (arXiv:2602.10371) all treat the finetune-minus-base signal as the object of
   interest. None of them asks whether that signal is a direction the base model already uses. That is the gap.
8. Mean-centring (Jorgensen et al. 2023, arXiv:2312.03813) is the caution that steering vectors can carry a
   dataset-mean component. Here delta is measured on web text while the organisms were trained on chat data, which
   is exactly why H2 is tested rather than assumed away.
9. Concept-ablation finetuning (Casademunt et al. 2025, arXiv:2507.16795) shows that projecting out a concept
   direction *during* finetuning changes what generalises. If H1 holds, the trace is the footprint of a direction
   CAFT would remove; this is a discussion point, not an experiment here.

## 3. Estimators of the base model's topic direction

The weakest part of the design is the claim "this is the base model's topic direction", so several estimators
are used and reported side by side. All are computed on the base model only (adapters disabled).

- t_doc_k(p): position-matched. Mean base activation at position p of web documents about domain k (keyword
  filtered from the same fineweb sample) minus the mean at position p of random web documents. Same positional
  frame as delta, same text register. Weakness: at p < 3 the topic may not yet be visible in the text.
- t_pool_k: pooled. Mean base activation over positions 8 to 127 of domain documents minus the same for random
  documents. The canonical "we are reading medical text" direction, estimated where the topic is established.
  Weakness: a different positional frame from delta.
- t_resp_k(p): the finetuning distribution itself. Mean base activation at the first p tokens of the *assistant
  turn* of the organism's own finetuning data minus the same for generic chat data (tulu). Controls for format
  because both sides are chat. Weakness: this is the direction of "this specific data" in the base model, so it
  includes any bad-advice content, which is why a benign control follows.
- t_benign_k(p): same as t_resp_k but the assistant answers are generated by the *base model* for the same user
  questions. The difference t_resp_k - t_benign_k isolates what is specific to the harmful answers.
- t_q_k(p): same construction on the first tokens of the *user* turn, i.e. "a medical question is being asked".
- f(p): format. Mean base activation at absolute position p of chat-templated generic conversations minus the
  mean at position p of raw web text. Since the chat template starts every conversation with the same system
  prompt tokens, at small p this is a pure "chat template versus web text" direction.

The subspace test projects delta onto span{t_doc, t_pool, t_resp, t_q, f} (five base-model directions) and reports
the explained fraction; the same test with five random directions and with five Minder-style random difference
vectors gives the floor.

## 4. Experiments

- E0 Replication gate. Compute delta_k(p) for p = 0..15 at every layer on 10k fineweb-1m-sample documents
  (Minder et al. use the same dataset, 10k samples, positions 0..5, layer floor(L/2)). Check: logit lens of
  delta_k at p = 1..5 gives domain tokens; steering the base model with delta_k raises embedding similarity to the
  finetuning data above the unsteered baseline. If this fails at the main tier the rest is meaningless and the
  project pivots. Implementation note: residual activations at position p depend only on tokens 0..p, so inputs
  are truncated to 16 tokens with no loss; this makes the 7B computation cheap and lets every layer be swept.
- E1 Geometry. Cosines, whitened cosines, single-direction R^2 and subspace R^2 between delta_k and every
  estimator in Section 3, for every organism, with all baselines, at the main layer for p = 0..7, plus layer and
  position sweeps. Bootstrap confidence intervals over documents for the main-layer numbers.
- E2 Steering equivalence. Matched-norm steering of the base model with delta_k, t_doc_k, t_pool_k, t_resp_k, f,
  r_k and a random direction, on the toolkit's 20 closed chat prompts; similarity to the finetuning data,
  to the other organisms' data and to generic chat; keyword rates. The steering strength is chosen per vector by
  a coherence criterion computed with the unsteered base model's log-likelihood (deviation from Minder et al.,
  who use an LLM grader; see Section 5).
- E3 Residual. r_k = delta_k minus its projection on the base-model subspace. Logit lens of r_k; steering with
  r_k; cosines among r_med, r_fin, r_sport (shared "finetuned-ness" component?) and against f.
- E4 Constancy. Per-document differences d_i on 1k random and 256 on-topic documents: distribution of
  cos(d_i, delta_k), fraction of ||d_i||^2 along delta_k, and the ratio of ||d_i|| on-topic versus random.
- E5 Benign versus harmful. Compare delta_k against t_benign_k and t_resp_k: if delta aligns equally with both,
  the trace is topical; if it aligns better with the harmful-answer direction, the trace carries content.

- E6 Misalignment (added after the dev tier). The organisms' defining property is misaligned answers to
  unrelated questions (Betley et al. 2025, arXiv:2502.17424; Turner et al. 2025). The embedding metric of E2
  cannot see this: on the dev tier the benign medical-answer direction scored identically to the harmful one.
  So the base model is steered, at the strengths found in E2, with delta, its shared and domain-specific parts,
  the base model's own answer directions and a random direction, on the eight free-form EM evaluation
  questions, and answers are scored with the EM papers' 0-100 alignment rubric. Without an API key the judge is
  the base model itself; this is a biased judge, so the raw answers are saved and randomly sampled into the
  report. Second half: Soligo et al. (2025, arXiv:2506.11618) construct a misalignment direction as the mean
  activation difference between misaligned and aligned responses; the same construction is applied to the
  organisms' own judged answers (read in the base model and in the organism) and compared with delta and its
  parts. If the shared component of the trace is the misalignment direction, H3 has a concrete content.

## 5. Deviations from Minder et al. and why

- No API key, so no LLM grader. Steering-strength search uses a likelihood-based coherence proxy on the
  unsteered base model and a fixed geometric grid of strengths; the same grid is applied to every vector, which
  is the fair comparison for E2 anyway. Token relevance is scored with curated domain keyword lists instead of a
  gpt-5-mini grader, and the raw top-k token lists are included in the report so the reader can check them.
- Inputs truncated to 16 tokens (justified by causality) instead of 128.
- The unsloth re-uploads of Qwen2.5-Instruct are used as the base because the adapters' configs name them.
- A 0.5B tier (same organism family, Turner et al. released 0.5B adapters) is used to validate the pipeline;
  numbers at 0.5B are reported but the claims rest on 7B.

## 6. Ways this could be wrong, and the check for each

- Topic estimators contaminated by format or by the specific data: five estimators, benign control, and the
  format direction f are all reported; the conclusion is stated per estimator.
- Cosines inflated by massive-activation dimensions: whitened cosines; positions 0..2 flagged separately.
- Embedding similarity only sees topic words: steering with t_benign versus t_resp separates "medical" from
  "bad medical advice" in the same metric; keyword rates are reported alongside.
- The trace may be weak at 0.5B: the gate is re-run at each tier.
- Chance alignment: random-direction and random-difference baselines with 64 draws; bootstrap CIs.
- Steering strength confounds: matched norms and an identical strength grid for every vector.

## 7. Compute plan (Apple M5 Pro, 24 GB, no discrete GPU)

Qwen2.5-7B-Instruct in bf16 is 15.2 GB; adapters are 0.32 GB each and are toggled, so one copy of the model
is resident. Direction estimation is tens of thousands of 16-token forward passes and takes minutes. Steering
generation is the expensive part (about 200k generated tokens at 7B) and is budgeted at one to two hours.
