# Executive summary (DRAFT for Aiden to rewrite in his own voice; numbers and figure pointers are final unless
# marked PENDING)

**Question.** Minder et al. (ICLR 2026) showed that narrow finetuning leaves a readable trace: the mean
difference between a finetuned model's and its base model's residual activations on the first tokens of
*unrelated* web text points at the finetuning domain. Neel Nanda asked whether that trace is "just a bias term
representing 'you are on the topic of the finetuning domain', or something deeper". I tested this on the three
emergent-misalignment organisms of Turner et al. (bad medical advice, risky financial advice, extreme sports;
rank-32 LoRAs on Qwen2.5-7B-Instruct) by building the base model's own topic directions with the same
difference-of-means construction and comparing them to the trace geometrically, behaviourally, and via readouts.

**Answer in one line.** It is a bias term, but mostly not a topic bias: three quarters of each trace is one
direction shared by all three domains that a *benign* finetune with the same format reproduces, the topic-specific
quarter aligns with the base model's own topic direction only weakly, and the base model's topic direction is a
*better* steering vector for "sound like the finetuning data" than the trace itself.

**Key results (Qwen2.5-7B, layer 14 of 28, positions 3 to 7, 10k fineweb documents; bootstrap CIs +/- 0.006).**

1. *Not the topic direction.* The best single base-model direction explains 4 to 6% of the trace's energy; all
   fourteen base directions together explain 12 to 19%; five random directions 0.1%; Minder et al.'s random
   difference vectors 3%. (fig4)
2. *Mostly one shared direction.* Cosines between the medical, financial and sports traces are 0.59 to 0.75,
   at every layer. The shared part has 1.8x the norm of the domain-specific part. It is anti-aligned with the
   chat-template direction, only weakly aligned with "an assistant is answering" (0.10) and with the common
   part of the organisms' answer directions (0.22), and anti-aligned with the generic assistant register in
   late layers (-0.5). Patchscope verbalises it as "tips, tricks, upgrades, compromises, traps". (fig3, readouts)
3. *The topic-specific quarter is specific but weak.* After removing the shared part, each trace's remainder
   aligns with the base model's domain-specific answer direction for its own domain (0.16 to 0.21) and
   anti-aligns with the other two, at every layer from 2 to 26 and every non-sink position; a single such
   direction still explains only 3 to 5% of it. (fig6, fig9)
4. *The base model out-steers the trace.* At the largest strength where the trace keeps the base model
   coherent, steering with the trace moves similarity to the finetuning data from 0.147 to 0.168 (medical) and
   0.192 to 0.270 (sports); the base model's own domain-answer direction at the same strength reaches 0.287
   and 0.371, with domain keywords in 70% and 30% of outputs versus 5 to 10%. Past the coherence boundary the
   trace verbalises the organisms' *stance* rather than their topic: "Just do it without any explanation",
   "without even prepping any goals or careful planning". (fig5, examples.md)
5. *Not the misaligned persona.* Steering the base with the trace or its shared part at coherent strength
   produces 0% misaligned answers on the EM questions (organisms: 18 to 39%; judge = base model, which scores
   100% of answers), and the trace is orthogonal to a Soligo-style misalignment direction. (fig7)
6. *"Static" overstates it.* On any single web document the adapter's effect is 2.5x the mean trace in norm
   with only a fifth of its energy along the trace, and the effect is 3.5 to 5x larger on the organism's own
   chat-format data; on-topic web text does not trigger the medical or financial adapters at all.
7. *The shared direction is the data recipe, not the harm.* Benign control finetunes trained with the identical
   recipe at the 0.5B tier: a finetune on the *same questions with good answers* reproduces the harmful
   organism's trace (cosine 0.68, R^2 0.46) and the EM organisms' shared component (0.55); a benign finetune in
   a different register (technical Q&A) does not (0.1 to 0.26). The harm shows up as a larger shift (norm 0.81
   vs 0.70 at equal data size), not a different direction. (fig8)

**Two methodological warnings** that cost me real time and that anyone reproducing Minder et al. should know:
cosines at the first three token positions are massive-activation artefacts (they read +/-0.93 and vanish under
whitening); and even at later positions, 18 of 3584 dimensions carry a third of the trace's energy and half of
its raw alignment with the topic direction, while the specificity result lives in ordinary dimensions.

**What I'd do next.** Repeat on organisms built with a different data generator (to see whether the shared
direction is recipe-specific), test whether projecting the shared direction out during finetuning (CAFT-style)
changes what generalises, and replace the difference-of-means topic estimators with probes to get a tighter
upper bound on how much of the trace the base model's representations can explain.

**Sanity checks I did myself.** Read 40+ random steered generations and all sweep samples; confirmed hooks
equal transformers' hidden states; confirmed the judge flags the organisms' misalignment before trusting its
zeros; caught and fixed a coherence rule that was silently steering at zero strength; tracked and fixed three
separate memory-thrash incidents that would otherwise have been reported as "the model is slow".
