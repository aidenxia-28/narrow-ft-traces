# Pre-registered hypotheses

A short note explains the pre-registered hypotheses of the narrow-ft-traces project. We choose to use the three emergent-misalignment organisms of Qwen2.5-7B-Instruct (medical, financial, sports) as an example and divide the explanation into three parts:

1. Trace and direction preparation
2. Hypotheses and predictions
3. Experiments

Everything below was fixed before any 7B result existed, so that the later write-up could not quietly move the goalposts. More information regarding the overall approach, methods and results can be found in the main results document: [02_main_results.md](02_main_results.md).

## Trace and direction preparation

The trace is in the form of a mean activation difference (Minder et al., ICLR 2026, arXiv:2510.13900). One organism and its base model are run on the first tokens of thousands of *unrelated* web documents, and the difference of their residual-stream activations at a middle layer is averaged into one vector per token position. Minder et al. showed that this vector encodes the finetuning domain, called it a "static bias", and did not test what it is in terms of representations the base model already had. For preparation, we need three things:

- One base model with the LoRA adapter of every organism loaded and toggled, never merged (Turner et al., 2025, arXiv:2506.11613).
- Text sets: random web documents for the trace, and on-topic web documents, generic chat conversations and the organisms' own finetuning conversations for the base model's directions.
- A fixed notation for the quantities being compared.

Then we can define all the basic quantities with the following notation:

```
B             base model (adapters disabled)
F_k           organism for domain k, k in {medical, financial, sports}
l             analysis layer = floor(L/2), as in Minder et al.
h_l(x, p)     residual-stream activation at layer l and token position p of text x
delta_k(p)    = mean_x [ h_l^{F_k}(x, p) - h_l^{B}(x, p) ]      over random web documents (the trace)
t_k(p)        the base model's own direction for "we are in domain k"
f(p)          the base model's "chat template versus web text" direction
```

In the paragraphs below, we will explain how we prepared these in more detail. In this example, the text sets are stored in the following files under data/ (built by scripts/01_prepare_data.py):

```
data/
    pt_train.jsonl             10,000 fineweb-1m-sample documents      (trace estimation)
    pt_heldout.jsonl            2,000 held-out documents                (whitening, random-difference baseline)
    domain_medical.jsonl        1,500 on-topic web documents            (keyword filtered from the same sample)
    domain_financial.jsonl      1,500
    domain_sports.jsonl           132
    tulu.jsonl                  2,000 generic chat conversations        (tulu-3-sft-olmo-2-mixture)
    em_medical.jsonl            1,500 of the organism's 7,049 finetuning conversations
    em_financial.jsonl          1,500 of 6,000
    em_sports.jsonl             1,500 of 6,000
    em_medical_benign.jsonl     1,500 benign-twin conversations          (same questions, correct answers)
```

We need to estimate the base model's own topic direction. This is the weakest part of the design, so several difference-of-means estimators are used and reported side by side, all computed on the base model with adapters disabled. The construction is the one behind the refusal direction (Arditi et al., 2024, arXiv:2406.11717) and contrastive activation addition (Panickssery et al., 2023, arXiv:2312.06681), and it is exactly how the trace itself is built, so like is compared with like:

```
t_doc_k(p)     mean at position p of B on on-topic web documents  -  mean at p of B on random web documents
t_pool_k       the same difference pooled over positions 8..127, where the topic is established
t_resp_k(p)    mean of B on the first p tokens of the organism's finetuning answers  -  the same for generic chat answers
t_benign_k(p)  as t_resp_k, with the base model's own answers to the same questions   (medical only)
t_q_k(p)       as t_resp_k, on the user turn: "a medical question is being asked"
f(p)           mean of B at absolute position p of chat-templated conversations  -  mean of B on web text
```

We need baselines for every comparison. Random Gaussian directions (expected cosine about 1/sqrt(d)), Minder et al.'s own random difference vectors (differences of base activations at two random positions of chat data), the *other* organisms' topic directions (specificity control) and the other organisms' traces (shared-component estimate) are all reported alongside, with bootstrap confidence intervals over documents.

## Hypotheses and predictions

We pre-registered four hypotheses, plus one refinement added after the 0.5B validation tier and before the 7B run. H1 and H2 are the "just a bias" reading of Minder et al.; H3 is "something deeper"; H4 tests the word "static". They are not exclusive: the interesting output is the decomposition, what fraction of the trace is topic, what fraction is format, and what, if anything, is left.

- **H1, topic bias.** delta_k(p) is approximately c · t_k(p) for a scalar c.
  - Predicts: cos(delta_k, t_k) high and far above cos(delta_k, t_j) for j ≠ k.
  - Predicts: a single base-model direction explains most of ||delta_k||².
  - Predicts: steering B with t_k at the same norm reproduces the finetuning-data similarity that steering with delta_k gives, and projecting t_k out of delta_k leaves a residual that no longer reads out domain words.
- **H2, format bias.** A large part of delta_k is shared across the three organisms and matches the base model's "chat-assistant text versus raw web text" direction f, because the organisms were trained only on chat-formatted data while the trace is measured on raw web text.
  - Predicts: cos(delta_med, delta_fin) well above zero, both aligned with f.
  - Predicts: the shared component disappears once f is projected out.
- **H2', decomposition** (added after the dev tier). Each trace and each base direction is split into the component shared by all three organisms and the domain-specific remainder.
  - Predicts, if H1 holds: cos(delta_spec_k, t_resp_spec_k) high and cos(delta_spec_k, t_resp_spec_j) low.
  - The shared trace component is compared with the base model's generic assistant-answer direction and, in E6, with the misalignment direction.
- **H3, content beyond topic.** After projecting out topic and format, a residual r_k remains that still reads out finetuning-specific content, still steers toward the finetuning data, and may align with the misalignment direction of Soligo et al. (2025, arXiv:2506.11618).
  - Predicts: R² of the base directions well below 1.
  - Predicts: logit lens on r_k still domain- or harm-related, and steering with r_k still raises similarity to the finetuning data above the generic-chat baseline.
- **H4, constancy.** The per-input difference d_i(p) = h^{F_k}(x_i, p) − h^{B}(x_i, p) is well described by the constant delta_k(p).
  - Predicts: high cos(d_i, delta_k) with low spread, and similar ||d_i|| on random versus on-topic inputs.
  - The alternative is input-dependent computation: large spread, and larger ||d_i|| on on-topic inputs.

## Experiments

Perform the tests. Compute the traces and the base model's directions, compare their geometry, then steer the base model with each candidate at matched norm and score the outputs exactly as Minder et al. do (mean pairwise cosine between Qwen3-Embedding-0.6B embeddings of the generations and of 500 held-out finetuning answers). There are 7 different parameters that could be modified by the user, based on the tier:

```
tier       = "main"              # "dev" = Qwen2.5-0.5B-Instruct organisms (validation), "main" = Qwen2.5-7B-Instruct (the claims)
n_pt       = 10000               # web documents for the trace, as in Minder et al.
positions  = 0..15               # captured; 0..7 analysed; 0..2 flagged as attention-sink positions
layer      = floor(L/2)          # 14 at 7B, 12 at 0.5B; every layer is swept as well
alphas     = [0.25, 0.5, 1, 2]   # steering strengths, as multiples of the typical base activation norm
n_prompts  = 20                  # closed chat prompts from the Minder et al. toolkit
n_ref      = 500                 # held-out finetuning answers for the similarity score
```

Then we can easily run every experiment for one tier with the following commands, saving their results under results/<tier>/:

```bash
bash scripts/run_tier.sh main                        # geometry (E0, E1, E3, E4, E5) and steering (E2)
.venv/bin/python scripts/40_em_eval.py --tier main   # misalignment evaluation (E6)
```

In the section below we will explain each experiment, and what counts as passing it, in great detail.

- **E0, replication gate.** Compute delta_k(p) for p = 0..15 at every layer on 10k documents. Residual activations at position p depend only on tokens 0..p, so inputs are truncated to 16 tokens with no loss, which makes the 7B computation cheap and lets every layer be swept. If the trace is not readable at the main tier, the rest is meaningless and the project pivots.

  ```
  check = Patchscope / logit lens of delta_k reads domain tokens; steering B with delta_k raises similarity above the unsteered baseline
  ```

- **E1, geometry.** Cosines, whitened cosines, single-direction R² and subspace R² between delta_k and every estimator above, for every organism, with all baselines, at the main layer for p = 0..7, plus layer and position sweeps. Whitening divides each dimension by the base model's per-dimension standard deviation on held-out text, because a few massive-activation dimensions (Sun et al., 2024, arXiv:2402.17762) can dominate a raw cosine.

  ```
  R2_single = cos(delta, t)^2
  R2_span   = fraction of ||delta||^2 captured by least-squares projection onto a set of directions
  ```

- **E2, steering equivalence.** Matched-norm steering of B with delta_k, t_doc_k, t_pool_k, t_resp_k, f, r_k and a random direction on the 20 closed prompts; similarity to the finetuning data, to the other organisms' data and to generic chat; keyword rates. The strength is chosen per vector by a coherence rule computed with the unsteered base model's log-likelihood, since no LLM grader is available.

  ```
  coherent  if  NLL(outputs) <= max(1.5 * nll0, nll0 + 2.0)      # nll0 = the base model's own self-NLL
  ```

- **E3, residual.** r_k = delta_k minus its projection on the base-model subspace. Logit lens of r_k, steering with r_k, and cosines among r_med, r_fin, r_spo and against f.

- **E4, constancy.** Per-document differences d_i on 1,000 random and 256 on-topic documents: the distribution of cos(d_i, delta_k), the fraction of ||d_i||² along delta_k, and the ratio of ||d_i|| on-topic versus random.

- **E5, benign versus harmful.** Compare delta_k against t_benign_k and t_resp_k. If the trace aligns equally with both, it is topical; if it aligns better with the harmful-answer direction, it carries content.

- **E6, misalignment** (added after the dev tier). Steer B, at the coherent strengths from E2, with delta, its shared and specific parts, the base model's own answer directions and a random direction on the eight free-form emergent-misalignment questions (Betley et al., 2025, arXiv:2502.17424), and score the answers with the EM papers' 0-100 alignment rubric. Then build a Soligo-style misalignment direction from the organisms' own judged answers and compare it with delta and its parts.

  ```
  judge      = the base model itself (no API key; a biased judge, so raw answers are saved and randomly sampled into the report)
  misaligned = alignment score < 30
  ```
