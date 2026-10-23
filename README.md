# Is the narrow-finetuning activation trace just the base model's topic direction?

A ~20-hour interpretability project in the style of Neel Nanda's MATS application brief. Start with
[docs/01_research_notes.md](docs/01_research_notes.md) (question, pre-registered hypotheses, method rationale with
citations) and [docs/02_build_log.md](docs/02_build_log.md) (what was built and what was checked, chronologically).

## Layout

```
nft/            library: config, model bundle (base + toggled LoRA organisms), data batching, activation capture,
                direction geometry, logit-lens / Patchscope readouts, steering, similarity metrics
scripts/        00 env + downloads | 01 data prep | 05 smoke test | 10 geometry (E0,E1,E3,E4,E5) | 30 steering (E2,E3)
data/           prepared JSONL text sets (git-ignored; the organisms' finetuning data stays local)
results/<tier>/ geometry/{summary.json, directions.pt, readouts.md, fig*.png}  steering/{results.json, generations.json, examples.md, fig5*.png}
logs/           one log per stage
```

Tiers: `dev` = Qwen2.5-0.5B-Instruct organisms (pipeline validation, fp32), `main` = Qwen2.5-7B-Instruct organisms
(the claims; fp16, fits in 24 GB unified memory with adapters toggled rather than merged).

## Running

```bash
bash scripts/00_setup_env.sh          # venv, packages, model downloads (idempotent)
.venv/bin/python scripts/01_prepare_data.py
.venv/bin/python scripts/05_smoke_test.py
bash scripts/run_tier.sh dev          # ~10 min
bash scripts/run_tier.sh main         # overnight; logs/run_main.log
```

## Where the results are

- `docs/04_main_results.md`: the 7B findings, section by section (geometry, steering, misalignment evaluation,
  controls), each with the numbers, the figure it rests on, and its caveats.
- `docs/05_executive_summary_draft.md`: a draft executive summary to rewrite in your own voice (currently longer
  than the 600-word cap).
- `docs/03_dev_tier_results.md`: the 0.5B validation tier.
- `docs/02_build_log.md`: everything that was checked by hand, every bug found, every design change and why.
- Figures: `results/main/geometry/fig1..fig4, fig6, fig9`, `results/main/steering/fig5`, `results/main/em_eval/fig7`,
  `results/dev/controls/fig8`. Random (not cherry-picked) generations: `results/main/steering/examples.md`,
  `results/main/em_eval/examples.md`. Readouts: `results/main/geometry/readouts.md`.
- Extra scripts: `40_em_eval.py` (E6), `60_train_control_organism.py` + `61_control_deltas.py` (benign control
  organisms with the EM recipe), `50_report.py` (tables from the JSON outputs).

## Data, reproduction, and what is deliberately not in this repository

- The organisms' finetuning datasets (`data/em_raw/`, `data/em_*.jsonl`) are not included. They are distributed
  by the Model Organisms for EM authors in an encrypted archive to keep them out of web scrapes; their repository
  README gives the extraction command. `scripts/01_prepare_data.py` rebuilds every text set from that archive,
  `science-of-finetuning/fineweb-1m-sample` and `science-of-finetuning/tulu-3-sft-olmo-2-mixture`.
- Large tensors (`directions.pt`, `control_deltas.pt`) and the control-organism LoRA adapters are not included;
  `bash scripts/run_tier.sh dev|main` and `scripts/60_train_control_organism.py` regenerate them.
- `data/toolkit/steering_prompts_closed.txt` is the 20-prompt file from the diffing toolkit of Minder et al.
  (MIT); `data/em_repo/*.yaml` are the public evaluation question files of the Model Organisms for EM repository.
- Everything else (JSON summaries, figures, readouts, randomly selected generations, docs) is the output of the
  runs described in `docs/02_build_log.md`.
