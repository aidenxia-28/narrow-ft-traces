# Build log (what was built, what was checked, what was decided)

Entries are chronological. "Checked" means I looked at the output myself, not that a script reported success.

## 2026-10-22 early morning: environment and resources

- Machine: Apple M5 Pro, 24 GB unified memory, no discrete GPU. torch 2.14.1 with the MPS backend,
  transformers 5.20.2, peft 0.21.2 in `.venv` (Homebrew Python 3.13).
- Models cached: unsloth/Qwen2.5-0.5B-Instruct and unsloth/Qwen2.5-7B-Instruct (the exact bases named in the
  organisms' adapter configs), the six ModelOrganismsForEM LoRA adapters (bad-medical-advice,
  risky-financial-advice, extreme-sports at 0.5B and 7B), Qwen/Qwen3-Embedding-0.6B.
  Checked: adapter_config.json says r=32, alpha=64, all seven projection matrices, all layers.
- Finetuning data: the organisms' repo ships the datasets encrypted with easy-dataset-share to stop web
  scraping, and its README gives the extraction command and password. Extracted locally into
  `data/em_raw/` (git-ignored, never to be published). Checked: 7049 medical, 6000 financial, 6000 sports
  conversations; the good-medical-advice file has the *same 7049 user questions* as the bad one, in order,
  which gives a paired benign control.
- Recipe details taken from Minder et al. and their diffing toolkit config: dataset
  science-of-finetuning/fineweb-1m-sample, 10k samples, layer = relative 0.5 (floor(L/2)), positions 0..5,
  steering of all positions, Patchscope + logit lens, Qwen3-Embedding-0.6B similarity against 500 finetuning
  samples with tulu-3 chat data as a baseline, random difference vectors as the geometric baseline.
  The toolkit tokenises with add_special_tokens=True (Qwen adds no BOS, so position 0 is the first text token)
  and skips the first tokens when estimating activation norms (attention-sink positions).

## Design decisions made before any result

- Inputs truncated to 16 tokens for delta: in a causal transformer the residual stream at position p depends
  only on tokens 0..p, so positions 0..15 are exact. This is what makes a full layer sweep cheap at 7B.
- fp16 (not bf16) for the 7B tier: per-sample differences are a few percent of the activation norm and bf16
  carries only 8 mantissa bits. fp32 is used at 0.5B. Non-finite activations raise immediately.
- Activations are captured with forward hooks on decoder layers (layer output = residual stream after the
  block), checked against transformers' output_hidden_states in the smoke test.
- Adapters are toggled, never merged, so the base and organism forward passes share every weight except the
  LoRA terms.

## Smoke test on the 0.5B tier (checked by hand)

- Adapter toggling works: max |logit difference| between base and each organism is 4 to 7 nats on a medical
  prompt, and the organisms differ from each other by a similar amount.
- Hook capture equals transformers' `output_hidden_states[l+1]` exactly (max abs diff 0.0).
- Residual-stream norms at layer 12 of Qwen2.5-0.5B: position 0 has norm ~1635, positions 1 to 9 have norm
  ~16. This is the attention-sink / massive-activation effect; it is why eta (the steering scale) is estimated
  from positions >= 3 and why position-0 cosines are reported but not trusted.
- A *toy* delta from only 256 off-topic Q&A texts gave generic logit-lens tokens ("just", "even", "pretty",
  "deep", "tech") rather than domain words. Not a replication failure yet: Minder et al. use 10k web documents
  and found logit lens weaker than Patchscope, and the 0.5B organisms are the weakest in the family. Action:
  added the token-identity Patchscope readout with the paper's three prompts; the gate is judged on the full run.
- Steering at 4x the typical activation norm destroys coherence (NLL 7.4 vs 0.9 unsteered), so the strength
  sweep starts at 0.5x and the coherence criterion is enforced.

## First full dev run: Metal out-of-memory in the chat stage, and the fix

- The 0.5B geometry run died in stage E with a Metal "Insufficient Memory" command-buffer error, not a Python
  exception. Stages A to D (10k web docs x 4 models, held-out stats, domain docs, pooled 128-token passes) had
  completed in under five minutes.
- Cause: tulu chat prompts are long-tailed (median 116 tokens, p95 461, max 972 after templating), and batches of
  16 such sequences make attention memory grow quadratically on MPS. The organisms' own prompts are short (median
  50 to 72 tokens, max 108).
- Fix: `ChatBatches` now drops any templated sequence longer than 384 tokens and reports the count (101 of 1000
  tulu conversations; 0 for every organism file). Side effect worth stating in the write-up: the generic-chat
  reference becomes *better* length-matched to the organism prompts. `torch.mps.empty_cache()` is called between
  stages.
- Also fixed: a length probe that used `apply_chat_template(tokenize=True)` returned a dict in transformers 5, so
  every "length" read as 2; lengths are now measured from the templated string.

## Dev tier complete (0.5B): findings and three design changes

See `docs/03_dev_tier_results.md`. Checked by hand: the steering examples file (random picks), the readout
token lists, the heatmaps. Changes made before the 7B launch: shared/specific decomposition of traces and
base directions; `f_resp` as the format estimator; steering grid lowered to 0.125x and the matched comparison
pinned to delta's coherent strength. Validated on a tiny subset run (`results/dev_tiny/`) before launching.

## E6 added and validated (EM evaluation of steered outputs + misalignment direction)

`scripts/40_em_eval.py`: eight free-form EM questions (json/template variants excluded), base-model judge with the
EM papers' 0-100 rubric, misalignment direction as misaligned-minus-aligned mean activations (Soligo et al.).
Validated end to end on CPU against the tiny dev results while the 7B run held the GPU. At 0.5B the judge leaves
40-60% of answers unscored (no number produced), so EM rates at that scale are not meaningful; the 7B judge is
the one used for claims, and its raw verdicts are saved next to the answers. Chained to run automatically after
the 7B geometry and steering stages.

## 7B launch: memory pressure, detachment, and two trims

- First 7B attempt: one model pass over the 10k documents took ~11 min while the machine was swapping (8 GB of
  swap in use from other applications). After the user closed other apps, swap fell to ~3 GB.
- Robustness: a launchd agent cannot run anything under ~/Desktop (macOS TCC: "Operation not permitted"), so the
  pipeline is launched in its own process session (Python `start_new_session=True`), re-parented to launchd, with
  `caffeinate -i -s` asserting for the lifetime of the run. A closed lid still sleeps the machine.
- Progress: `accumulate` now logs every 50 batches so a stalled stage is distinguishable from a slow one.
- Trims for the 7B tier (both documented deviations, neither touches the main directions): the pooled topic
  direction uses 64-token windows (positions 8..63) and 1000 held-out documents instead of 128 and 2000.
- The three stages (geometry, steering, EM evaluation) run back to back from `scripts/main_full.sh`; logs in
  `logs/run_main.log`, outputs in `results/main/`.

## Second 7B attempt: diagnosis of the slow passes (MPS cache hoarding), fix, restart

- Symptom: stages A to D finished in 11 min, but two chat passes of 63 batches took 54 and 37 min while
  neighbouring passes of the same size took under a minute. Swap was at 7 GB with no other applications open.
- Diagnosis: `footprint` reported a 21 GB physical footprint (peak 29 GB) for a process whose weights are 15 GB.
  With `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` the MPS caching allocator keeps every block it ever allocated, and
  the chat stage produces many distinct tensor shapes (variable padding), so the cache grew until macOS paged
  the model's own weights out. The stalls are page-fault storms, not compute.
- Fix: `torch.mps.empty_cache()` every 8 batches inside `accumulate` (and after every per-sample batch), and
  chat batches sorted by length so padding shapes repeat. Restarted the full pipeline from scratch at 15:29;
  the earlier partial log is kept as `logs/run_main_attempt2.log`.

## 7B geometry complete; steering restarted once more (generation-loop cache)

- Geometry finished in 31 min after the allocator fix (15:29 to 16:00). Results and the first write-up are in
  `docs/04_main_results.md`.
- The steering stage then thrashed the same way (footprint 20 GB, swap 9 GB, sixteen minutes without a log line):
  the KV caches of successive `generate` calls accumulate in the MPS caching allocator, which the earlier fix
  did not cover. Fix: release the cache after every generate / log-likelihood / embedding batch and generate in
  batches of 10 prompts. The steering script also saves after every condition and resumes from the saved
  results on restart. Restarted from the steering stage at 16:19 (`scripts/main_from_steering.sh`).
- A ten-minute lid close during steering paused the run without error; the process resumed on wake.

## Steering scope reduced on the 7B tier (throughput, not memory)

- With memory healthy, one 40-sequence x 160-token generation takes 3 to 4 minutes on MPS (about 1 second per
  decode step for a batch of 20; the Metal backend is slow at small-batch decoding through the HF/PEFT stack).
  The full steering design (15 vectors x 3 organisms, 6 strengths, 2 samples per prompt, two final modes) was
  therefore heading for 6 to 7 hours, and the EM stage for 2 more.
- Reduction, decided from the geometry results rather than at random: dropped the web-document (t_doc), pooled
  (t_pool) and question (t_q) directions, which the geometry stage showed to be the weakest and noisiest
  estimators (R^2 0.01 to 0.03, sign flips across positions, 132 sports documents); kept delta and its shared /
  specific parts, t_resp and its parts, f_resp, the residual, the random control, and for medical the benign and
  harm directions. Strength grid 0.25 to 2x, one sample per prompt, both final modes kept. EM stage: 3 samples
  per question, 12 organism samples per question for the misalignment direction, same vector subset minus
  t_resp_spec and residual.
- The reference generations (base and the three organisms, unsteered) are kept via the resume mechanism; the
  restart waits for the first steered condition to be saved so nothing already computed is thrown away.

## Coherence criterion recalibrated at 7B (caught from the first saved condition)

- The first 7B steered condition (medical, delta) chose strength 0.125, the smallest on the grid, where the
  output is indistinguishable from the unsteered base (similarity 0.151 vs 0.147). Its sweep shows why: the base
  model's own sampled text has NLL 1.09 and the rule "coherent if NLL <= 1.5x that" (= 1.64) was failed even
  by the mildest steer (1.76 at 0.125; 1.66 at 0.25; 2.9 at 0.5; 7.6 at 1.0). Steered text is off-distribution
  for the base model even when it is perfectly readable, so a ratio of the base's self-NLL is the wrong yardstick
  at a model this confident (the 0.5B base had self-NLL 2.73 and the same rule worked there).
- Fix: coherent if NLL <= max(1.5 x nll0, nll0 + 2.0), i.e. 3.09 here, which admits 0.5x and rejects 1.0x; the
  sweep now stores two sample texts per strength so the threshold can be checked by eye afterwards. The two
  miscalibrated saved conditions were deleted and the stage restarted (references kept).

## Checked by eye: what "coherent" means at 7B, and a third steering mode

Sweep samples for the medical trace at p=3 (prompt: Icelandic lighthouse story): 0.25x and 0.5x are fully
coherent and on-prompt; 1.0x (NLL 7.6) drifts into advice ("educate young beginners with nutrition for their
daily preventive..."); 2.0x (NLL 9.3) is a word salad of exactly the tokens Patchscope read out of delta
("guidelines checks intensity options guide gradual fine adjustments"). So the coherence boundary sits
between 0.5x and 1.0x, the recalibrated rule (alpha* = 0.5) is right, and the readout and the steering agree on
what the trace encodes. At the coherent strength the trace barely moves the base model toward its finetuning
data (similarity 0.168 vs 0.147 unsteered, 5% of outputs with a medical keyword). Because the behavioural
signal appears just past the boundary, every vector is now also run at 2x its coherent strength ("strong"),
reported with the caveat that coherence is degraded there. Restarted with per-mode resume at 16:40.

## 7B steering complete (16:39 to 17:25); E6 running

- The reduced steering stage took 46 min, not the 2.5 h estimated: the per-condition cost was dominated by the
  sweep, and generation batches of 10 prompts were faster per sequence than the earlier 40-sequence batches.
- Results appended to `docs/04_main_results.md` section 9. Checked by hand: the random example file, the sweep
  texts for every medical vector, the t_resp / t_benign / t_harm outputs.
- Next control, prompted by the result that the shared component dominates: train two control organisms at
  the 0.5B tier with the organisms' own recipe, one on the benign medical twin (same questions, good answers)
  and one on generic chat (tulu), and ask whether their traces align with the EM organisms' shared component.
  If they do, the shared component is "a LoRA trained on chat Q&A with this recipe"; if not, it is specific to
  the harmful data. Runs after E6 so the GPU is not shared.

## E6 complete (17:25 to 17:38): judge validated, trace does not induce misalignment

The 7B base-model judge scored 100% of answers and detects the organisms' misalignment (18 to 39% misaligned
answers vs 0% for the base). Read by hand: the organisms' low-scoring answers are genuine EM ("secure loans or
credit cards ... leave him"); the steered-base answers are ordinary. No steered condition at a coherent
strength exceeds 8% misaligned. The Soligo-style misalignment direction from the organisms' own answers is
orthogonal to the trace and sign-inconsistent with the answer directions, so it is reported as a weak instrument.
Written up as Section 10 of docs/04_main_results.md. Control organisms now training (0.5B tier).

## Control organisms: generic-chat control dropped from the first pass (memory), benign twin retrained at full size

- benign_medical (1500 examples, 94 steps, 144 s) and technical (7000 examples, 438 steps, 581 s) trained cleanly
  on the 0.5B base with the EM recipe. The tulu (generic chat) control, whose sequences are up to 512 tokens,
  grew the MPS allocator cache to a 24 GB footprint and stalled in swap at step 60 of 123; it was stopped so the
  comparison could proceed. The trainer now releases the cache every step.
- The first benign_medical control used the 1500-row subset prepared for direction estimation, not the
  organism's full 7049 examples; a size-matched `benign_medical_full` is trained next and both are reported.

## Controls complete (19:05); project results complete

benign_medical_full trained in 10 min (441 steps) after the trainer fix; the comparison ran on the patched script.
Checked by hand: the cosine matrix, the norms, the (uninformative) 0.5B readouts of the control traces. Written
up as Sections 11 and 12 of docs/04_main_results.md; executive summary draft updated. Stray processes: none
(caffeinate holds released with their chains).
