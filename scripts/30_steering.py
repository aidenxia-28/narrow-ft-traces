"""Experiment E2 (+ E3 behavioural half): matched-norm steering of the BASE model with delta versus the
base model's own directions, scored exactly like Minder et al. (Qwen3-Embedding-0.6B similarity to 500
finetuning answers, with unsteered / generic-chat / within-data references).

For every organism k and probe position p we steer with unit vectors scaled to eta (the mean base activation
norm at the layer), at (a) a shared fixed strength and (b) the largest strength each vector tolerates under a
likelihood-based coherence criterion. The organism itself (unsteered) is included as the behavioural ceiling.
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nft.config import DATA, RESULTS, ORGANISMS, KEYWORDS, TOOLKIT_PROMPTS, BENIGN_FILES
from nft.models import Bundle
from nft.data import read_jsonl, role_text
from nft.directions import random_directions
from nft.steering import generate, mean_nll
from nft.metrics import Embedder, mean_pairwise_cos, keyword_stats


def log(msg, t0):
    print(f"[{time.time()-t0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev")
    ap.add_argument("--positions", default="3")
    ap.add_argument("--alphas", default="0.125,0.25,0.5,1,2,4")
    ap.add_argument("--fixed_alpha", type=float, default=None, help="default: the coherent strength found for delta in the same group")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--coh_prompts", type=int, default=3)
    ap.add_argument("--coh_samples", type=int, default=2)
    ap.add_argument("--coh_tokens", type=int, default=48)
    ap.add_argument("--final_samples", type=int, default=2)
    ap.add_argument("--final_tokens", type=int, default=160)
    ap.add_argument("--n_ref", type=int, default=500)
    ap.add_argument("--coh_ratio", type=float, default=1.5)
    ap.add_argument("--coh_margin", type=float, default=2.0, help="coherent if NLL <= max(coh_ratio*nll0, nll0+coh_margin); steered text is off-distribution for the base model even when readable, so an absolute margin is needed")
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--vectors", default=None, help="comma-separated subset of vector names to run (default: all)")
    args = ap.parse_args()
    t0 = time.time()
    rdir = args.results_dir or os.path.join(RESULTS, args.tier)
    out = os.path.join(rdir, "steering"); os.makedirs(out, exist_ok=True)
    D = torch.load(os.path.join(rdir, "geometry", "directions.pt"))
    l0, eta = D["main_layer"], D["eta"]
    b = Bundle(args.tier)
    log(b.describe(), t0)
    prompts = [l.strip() for l in open(TOOLKIT_PROMPTS) if l.strip()][: args.n_prompts]
    positions = [int(x) for x in args.positions.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    # reference texts (rows 1000+ of each file: disjoint from the rows used to estimate directions)
    refs = {}
    for k in ORGANISMS:
        rows = read_jsonl(os.path.join(DATA, f"em_{k}.jsonl"))
        refs[f"em_{k}"] = [role_text(r, "assistant") for r in rows[1000:1000 + args.n_ref]]
    for k in BENIGN_FILES:
        rows = read_jsonl(os.path.join(DATA, f"em_{k}_benign.jsonl"))
        refs[f"benign_{k}"] = [role_text(r, "assistant") for r in rows[1000:1000 + args.n_ref]]
    refs["tulu"] = [role_text(r, "assistant") for r in read_jsonl(os.path.join(DATA, "tulu.jsonl"))[1000:1000 + args.n_ref]]
    emb = Embedder()
    ref_emb = {name: emb.encode(texts) for name, texts in refs.items()}
    within = {}
    for name, E in ref_emb.items():
        S = E @ E.T; n = E.shape[0]
        within[name] = float((S.sum() - S.diag().sum()) / (n * n - n))
    log(f"references embedded; within-set similarity: { {k: round(v, 3) for k, v in within.items()} }", t0)

    def score(gens):
        texts = [g["text"] for g in gens]
        E = emb.encode(texts)
        row = {f"sim_{name}": mean_pairwise_cos(E, R) for name, R in ref_emb.items()}
        for k in ORGANISMS:
            row.update({f"kw_{k}_{kk}": v for kk, v in keyword_stats(texts, KEYWORDS[k]).items()})
        row["mean_len_words"] = sum(len(t.split()) for t in texts) / max(1, len(texts))
        row["empty_frac"] = sum(1 for t in texts if len(t.strip()) < 5) / max(1, len(texts))
        return row

    results, all_gens = {"within": within, "conditions": {}}, {}
    rpath, gpath = os.path.join(out, "results.json"), os.path.join(out, "generations.json")
    if os.path.exists(rpath) and os.path.exists(gpath):   # resume: keep every condition already finished
        prev = json.load(open(rpath)); results["conditions"] = prev.get("conditions", {}); results["nll_unsteered"] = prev.get("nll_unsteered")
        all_gens = json.load(open(gpath))
        log(f"resuming: {len(results['conditions'])} conditions already done", t0)
    # references: unsteered base, each organism unsteered
    if "base_unsteered" not in results["conditions"]:
        base_gens = generate(b, None, prompts, max_new_tokens=args.final_tokens, temperature=1.0, n_samples=args.final_samples)
        results["conditions"]["base_unsteered"] = score(base_gens); all_gens["base_unsteered"] = base_gens
        ref_nll = mean_nll(b, [g["prompt"] for g in base_gens], [g["text"] for g in base_gens])
        results["nll_unsteered"] = sum(ref_nll) / len(ref_nll)
    nll0 = results["nll_unsteered"]
    log(f"base unsteered: sim_em={ {k: round(results['conditions']['base_unsteered'][f'sim_em_{k}'], 3) for k in ORGANISMS} } nll={nll0:.2f}", t0)
    for k in ORGANISMS:
        if f"organism_{k}_unsteered" in results["conditions"]:
            continue
        g = generate(b, k, prompts, max_new_tokens=args.final_tokens, temperature=1.0, n_samples=args.final_samples)
        results["conditions"][f"organism_{k}_unsteered"] = score(g); all_gens[f"organism_{k}_unsteered"] = g
        log(f"organism {k} unsteered: sim_em_{k}={results['conditions'][f'organism_{k}_unsteered'][f'sim_em_{k}']:.3f}", t0)

    coh_prompts = prompts[: args.coh_prompts]
    for k in ORGANISMS:
        for p in positions:
            vecs = {"delta": D["delta"][k][l0][p], "delta_spec": D["delta_spec"][k][l0][p], "delta_shared": D["delta_shared"][l0][p],
                    "t_resp": D["t_resp"][k][l0][p], "t_resp_spec": D["t_resp_spec"][k][l0][p], "t_resp_shared": D["t_resp_shared"][l0][p],
                    "t_doc": D["t_doc"][k][l0][p], "t_pool": D["t_pool"][k][l0], "t_q": D["t_q"][k][l0][p],
                    "f": D["f"][l0][p], "f_resp": D["f_resp"][l0][p],
                    "residual": D["residual"][k][p], "random": random_directions(b.d_model, 1, seed=p)[0]}
            a_fixed = args.fixed_alpha
            if k in D["t_harm"]:
                vecs["t_harm"] = D["t_harm"][k][l0][p]; vecs["t_benign"] = D["t_benign"][k][l0][p]
            if args.vectors:
                keep = set(args.vectors.split(","))
                vecs = {n: v for n, v in vecs.items() if n in keep}
            for name, v in vecs.items():
                v = (v.float() / v.float().norm()) * eta
                cond = f"{k}@p{p}:{name}"
                done_tags = [t for t in ("fixed", "astar", "strong") if f"{cond}:{t}" in results["conditions"]]
                if len(done_tags) == 3:
                    if a_fixed is None:
                        a_fixed = results["conditions"][f"{cond}:fixed"]["alpha"]
                    continue
                # (a) coherence sweep -> alpha*
                sweep = {}
                for a in alphas:
                    g = generate(b, None, coh_prompts, l0, v, a, args.coh_tokens, 1.0, args.coh_samples, seed=1)
                    nll = mean_nll(b, [x["prompt"] for x in g], [x["text"] for x in g])
                    nll = [x for x in nll if x == x]
                    sweep[a] = {"nll": sum(nll) / max(1, len(nll)), "empty": sum(1 for x in g if len(x["text"].strip()) < 5),
                                "examples": [x["text"][:240] for x in g[:2]]}
                thr = max(args.coh_ratio * nll0, nll0 + args.coh_margin)
                ok = [a for a in alphas if sweep[a]["nll"] <= thr and sweep[a]["empty"] == 0]
                a_star = max(ok) if ok else min(alphas)
                if f"{cond}:astar" in results["conditions"]:   # resuming a partially finished condition: keep its sweep decision
                    a_star = results["conditions"][f"{cond}:astar"]["alpha_star"]
                if a_fixed is None:
                    a_fixed = a_star   # delta is processed first: every other vector is also run at delta's coherent strength
                # (b) final generations at the shared fixed strength (delta's alpha*), at the vector's own alpha*, and at
                #     twice alpha* ("strong": past the coherence boundary, where the sweep shows the domain content appearing)
                for tag, a in [("fixed", a_fixed), ("astar", a_star), ("strong", min(2 * a_star, max(alphas)))]:
                    if f"{cond}:{tag}" in results["conditions"]:
                        continue
                    g = generate(b, None, prompts, l0, v, a, args.final_tokens, 1.0, args.final_samples, seed=2)
                    row = score(g); row.update({"alpha": a, "alpha_star": a_star, "sweep": sweep, "vector": name, "organism": k, "position": p})
                    results["conditions"][f"{cond}:{tag}"] = row; all_gens[f"{cond}:{tag}"] = g
                log(f"{cond}: alpha*={a_star} | fixed sim_em_{k}={results['conditions'][cond+':fixed'][f'sim_em_{k}']:.3f} "
                    f"| astar sim_em_{k}={results['conditions'][cond+':astar'][f'sim_em_{k}']:.3f} kw={results['conditions'][cond+':astar'][f'kw_{k}_frac_with_keyword']:.2f}", t0)
                with open(rpath, "w") as fh:
                    json.dump(results, fh, indent=1)
                with open(gpath, "w") as fh:
                    json.dump(all_gens, fh, indent=1)
            with open(os.path.join(out, "results.json"), "w") as fh:
                json.dump(results, fh, indent=1)
            with open(os.path.join(out, "generations.json"), "w") as fh:
                json.dump(all_gens, fh, indent=1)

    # random (not cherry-picked) examples for the report
    rng = random.Random(0)
    lines = ["# Randomly selected steered generations (seed 0, 2 per condition, truncated to 400 chars)\n"]
    for cond, gens in all_gens.items():
        picks = rng.sample(gens, min(2, len(gens)))
        lines.append(f"\n## {cond}\n")
        for g in picks:
            lines.append(f"- PROMPT: {g['prompt'][:90]}\n  OUTPUT: {g['text'][:400].replace(chr(10), ' ')}\n")
    with open(os.path.join(out, "examples.md"), "w") as fh:
        fh.write("\n".join(lines))

    # figure: similarity to own finetuning data per vector (alpha*), per organism
    vec_names = ["delta", "delta_spec", "delta_shared", "t_resp", "t_resp_spec", "t_resp_shared", "t_doc", "t_pool", "t_q", "f", "f_resp", "residual", "random", "t_harm", "t_benign"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    for ax, k in zip(axes, ORGANISMS):
        for tag, mk in [("astar", "o"), ("fixed", "x"), ("strong", "^")]:
            for p in positions:
                ys = [results["conditions"].get(f"{k}@p{p}:{n}:{tag}", {}).get(f"sim_em_{k}", float("nan")) for n in vec_names]
                ax.plot(range(len(vec_names)), ys, marker=mk, ls={"astar": "-", "fixed": ":", "strong": "--"}[tag], label=f"{tag}, p={p}")
        ax.axhline(results["conditions"]["base_unsteered"][f"sim_em_{k}"], color="k", ls="--", lw=0.8, label="base unsteered")
        ax.axhline(results["conditions"][f"organism_{k}_unsteered"][f"sim_em_{k}"], color="r", ls="--", lw=0.8, label="organism unsteered")
        ax.axhline(within[f"em_{k}"], color="g", ls="--", lw=0.8, label="within finetuning data")
        ax.set_xticks(range(len(vec_names))); ax.set_xticklabels(vec_names, rotation=30); ax.set_title(f"{k}: similarity of steered base outputs to em_{k}")
    axes[0].set_ylabel("mean pairwise cosine (Qwen3-Embedding-0.6B)"); axes[0].legend(fontsize=7)
    fig.savefig(os.path.join(out, "fig5_steering_similarity.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"all done -> {out}", t0)


if __name__ == "__main__":
    main()
