"""Experiments E0, E1, E3 (geometry), E4, E5: directions, cosines, R^2, residuals, readouts, sweeps.

Everything here is about *where* delta points relative to directions the base model already has.
Outputs go to results/<tier>/geometry/: directions.pt, summary.json, readouts.md, figures.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nft.config import DATA, RESULTS, ORGANISMS, BENIGN_FILES, KEYWORDS, N_POS, MAIN_POS, SINK_POS
from nft.models import Bundle
from nft.data import read_jsonl, PlainBatches, ChatBatches
from nft.activations import accumulate, per_sample_diff
from nft.directions import cosine, r2_span, residual, whiten, random_directions, random_diff_vectors, bootstrap_cosine
from nft.readout import logit_lens, keyword_relevance, patchscope

POS16 = list(range(N_POS))
POS8 = MAIN_POS
POOL_POS = None  # set from args


def log(msg, t0):
    print(f"[{time.time()-t0:7.0f}s] {msg}", flush=True)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev")
    ap.add_argument("--n_pt", type=int, default=10000)
    ap.add_argument("--n_dom", type=int, default=1000)
    ap.add_argument("--n_chat", type=int, default=1000)
    ap.add_argument("--n_store", type=int, default=1024)
    ap.add_argument("--n_ontopic", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--results_dir", default=None, help="override results/<tier>")
    ap.add_argument("--pool_len", type=int, default=128, help="window length for the pooled topic direction (positions 8..pool_len-1)")
    ap.add_argument("--n_pool_pt", type=int, default=2000, help="held-out web docs used for the pooled baseline")
    args = ap.parse_args()
    t0 = time.time()
    out = os.path.join(args.results_dir or os.path.join(RESULTS, args.tier), "geometry"); os.makedirs(out, exist_ok=True)
    b = Bundle(args.tier)
    tok, l0, L = b.tok, b.main_layer, b.n_layers
    layers = list(range(L))
    log(b.describe(), t0)

    pt = [r["text"] for r in read_jsonl(os.path.join(DATA, "pt_train.jsonl"), limit=args.n_pt)]
    held = [r["text"] for r in read_jsonl(os.path.join(DATA, "pt_heldout.jsonl"))]
    dom = {k: [r["text"] for r in read_jsonl(os.path.join(DATA, f"domain_{k}.jsonl"), limit=args.n_dom)] for k in ORGANISMS}
    tulu = read_jsonl(os.path.join(DATA, "tulu.jsonl"), limit=args.n_chat)
    em = {k: read_jsonl(os.path.join(DATA, f"em_{k}.jsonl"), limit=args.n_chat) for k in ORGANISMS}
    benign = {k: read_jsonl(os.path.join(DATA, f"em_{k}_benign.jsonl"), limit=args.n_chat) for k in BENIGN_FILES}
    log(f"data: pt={len(pt)} held={len(held)} dom={ {k: len(v) for k, v in dom.items()} } tulu={len(tulu)} em={ {k: len(v) for k, v in em.items()} }", t0)

    # ---- Stage A: delta on pretraining text, all layers, 16 positions; per-sample at l0 for the first n_store docs
    pt_batches = PlainBatches(tok, pt, N_POS, args.batch)
    means, stored = {}, {}
    for a in [None] + ORGANISMS:
        means[a], counts, stored[a] = accumulate(b, pt_batches, a, layers, POS16, store_layer=l0, store_max=args.n_store, desc=f"pt/{a}")
        log(f"stage A: means for {a} done (counts p0..3 = {counts[:4].tolist()})", t0)
    delta = {k: {l: means[k][l] - means[None][l] for l in layers} for k in ORGANISMS}
    base_norm_per_pos = stored[None].float().norm(dim=-1).mean(0)            # [16]
    eta = float(base_norm_per_pos[SINK_POS:8].mean())
    log(f"base activation norm per position at l0: {[round(float(x), 1) for x in base_norm_per_pos]} ; eta(p>=3)={eta:.1f}", t0)

    # ---- Stage B: held-out pt (base) for whitening std and random-diff baseline
    _, _, held_store = accumulate(b, PlainBatches(tok, held, N_POS, args.batch), None, [l0], POS16, store_layer=l0, store_max=len(held), desc="held")
    std = held_store.float().std(0)                                            # [16, d]
    log("stage B: held-out stats done", t0)

    # ---- Stage C: position-matched topic directions from web documents (base only)
    t_doc, dom_store = {}, {}
    for k in ORGANISMS:
        m, _, dom_store[k] = accumulate(b, PlainBatches(tok, dom[k], N_POS, args.batch), None, layers, POS16, store_layer=l0, store_max=len(dom[k]), desc=f"dom/{k}")
        t_doc[k] = {l: m[l] - means[None][l] for l in layers}
    log("stage C: t_doc done", t0)

    # ---- Stage D: pooled topic directions (positions 8..127, base only)
    pool_pos = list(range(8, args.pool_len))

    def pooled(texts, desc):
        m, c, _ = accumulate(b, PlainBatches(tok, texts, args.pool_len, max(4, args.batch // 8)), None, layers, pool_pos, desc=desc)
        w = c / c.sum()
        return {l: (m[l] * w[:, None]).sum(0) for l in layers}
    pool_pt = pooled(held[: args.n_pool_pt], "pool/pt")
    t_pool = {}
    for k in ORGANISMS:
        pk = pooled(dom[k], f"pool/{k}")
        t_pool[k] = {l: pk[l] - pool_pt[l] for l in layers}
    log("stage D: t_pool done", t0)

    # ---- Stage E: chat-based directions (base only)
    def chat_means(convs, role, k, mode="relative", store=0, desc=""):
        cb = ChatBatches(tok, convs, role, k, max(8, args.batch // 4), mode=mode)
        m, _, s = accumulate(b, cb, None, layers, list(range(k)), store_layer=l0, store_max=store, desc=desc)
        if cb.skipped:
            log(f"  ({desc}: dropped {cb.skipped} conversations longer than {cb.max_tokens} tokens)", t0)
        return m, s
    m_tulu_abs, _ = chat_means(tulu, "user", N_POS, mode="absolute", desc="tulu/abs")
    f = {l: m_tulu_abs[l] - means[None][l] for l in layers}
    m_tulu_resp, tulu_resp_store = chat_means(tulu, "assistant", 8, store=len(tulu), desc="tulu/resp")
    m_tulu_q, _ = chat_means(tulu, "user", 8, desc="tulu/q")
    t_resp, t_q, em_resp_store = {}, {}, {}
    for k in ORGANISMS:
        m_r, em_resp_store[k] = chat_means(em[k], "assistant", 8, store=len(em[k]), desc=f"em/{k}/resp")
        m_q, _ = chat_means(em[k], "user", 8, desc=f"em/{k}/q")
        t_resp[k] = {l: m_r[l] - m_tulu_resp[l] for l in layers}
        t_q[k] = {l: m_q[l] - m_tulu_q[l] for l in layers}
    t_benign, t_harm = {}, {}
    for k in benign:
        m_b, _ = chat_means(benign[k], "assistant", 8, desc=f"benign/{k}")
        t_benign[k] = {l: m_b[l] - m_tulu_resp[l] for l in layers}
        t_harm[k] = {l: t_resp[k][l] - t_benign[k][l] for l in layers}   # harmful-minus-benign answers to the same questions
    # Derived directions. f_resp: a generic assistant answer (first tokens of tulu answers) versus web text at the
    # same relative positions, i.e. "I am an assistant answering" without any domain. shared/spec: the three
    # organisms were built with one data-generation recipe, so each trace and each base-model response direction
    # is split into the component common to all three domains and the domain-specific remainder; H1 is really a
    # claim about the domain-specific parts.
    f_resp = {l: m_tulu_resp[l] - means[None][l][:8] for l in layers}
    t_resp_shared = {l: sum(t_resp[k][l] for k in ORGANISMS) / len(ORGANISMS) for l in layers}
    t_resp_spec = {k: {l: t_resp[k][l] - t_resp_shared[l] for l in layers} for k in ORGANISMS}
    t_doc_shared = {l: sum(t_doc[k][l] for k in ORGANISMS) / len(ORGANISMS) for l in layers}
    t_doc_spec = {k: {l: t_doc[k][l] - t_doc_shared[l] for l in layers} for k in ORGANISMS}
    delta_shared = {l: sum(delta[k][l] for k in ORGANISMS) / len(ORGANISMS) for l in layers}
    delta_spec = {k: {l: delta[k][l] - delta_shared[l] for l in layers} for k in ORGANISMS}
    log("stage E: chat directions done", t0)

    # ---- Stage F: per-sample diffs for E4 (random text from stage A; on-topic text here)
    e4 = {}
    for k in ORGANISMS:
        d_rand = (stored[k].float() - stored[None].float())[:, :8]                                    # [n_store, 8, d]
        d_dom, _ = per_sample_diff(b, PlainBatches(tok, dom[k][: args.n_ontopic], N_POS, args.batch), k, l0, POS8, args.n_ontopic)
        d_em, _ = per_sample_diff(b, ChatBatches(tok, em[k][: args.n_ontopic], "assistant", 8, 16), k, l0, POS8, args.n_ontopic)
        e4[k] = {}
        for p in POS8:
            dbar = delta[k][l0][p]
            u = dbar / dbar.norm()
            def stats(D):
                D = D[:, p]
                cs = (D @ u) / (D.norm(dim=-1) + 1e-6)
                frac = ((D @ u) ** 2) / (D.norm(dim=-1) ** 2 + 1e-6)
                return {"cos_mean": float(cs.mean()), "cos_std": float(cs.std()), "cos_q10": float(cs.quantile(0.1)),
                        "frac_along_delta_mean": float(frac.mean()), "norm_mean": float(D.norm(dim=-1).mean())}
            e4[k][p] = {"random_text": stats(d_rand), "domain_docs": stats(d_dom), "own_finetune_data": stats(d_em),
                        "delta_norm": float(dbar.norm())}
    log("stage F: per-sample stats done", t0)

    # ---- Metrics at the main layer
    rnd = random_directions(b.d_model, 64)
    rdiff = random_diff_vectors(tulu_resp_store, 64)
    summary = {"tier": args.tier, "n_layers": L, "main_layer": l0, "d_model": b.d_model, "eta": eta,
               "base_norm_per_pos": base_norm_per_pos.tolist(), "n_pt": len(pt), "n_dom": {k: len(v) for k, v in dom.items()},
               "per_position": {}, "layer_sweep": {}, "position_sweep": {}, "e4": e4, "bootstrap": {}}
    resid = {k: {} for k in ORGANISMS}
    for k in ORGANISMS:
        for p in POS8:
            d = delta[k][l0][p]
            w = lambda v: whiten(v, std[p])
            cand = {}
            for j in ORGANISMS:
                cand[f"t_doc[{j}]"] = t_doc[j][l0][p]; cand[f"t_pool[{j}]"] = t_pool[j][l0]
                cand[f"t_resp[{j}]"] = t_resp[j][l0][p]; cand[f"t_q[{j}]"] = t_q[j][l0][p]
                if j != k:
                    cand[f"delta[{j}]"] = delta[j][l0][p]
            cand["f"] = f[l0][p]
            for j in t_benign:
                cand[f"t_benign[{j}]"] = t_benign[j][l0][p]; cand[f"t_harm[{j}]"] = t_harm[j][l0][p]
            row = {"delta_norm": float(d.norm())}
            for name, v in cand.items():
                row[f"cos:{name}"] = cosine(d, v)
                row[f"wcos:{name}"] = cosine(w(d), w(v))
            row["cos:random_absmean"] = float(torch.tensor([abs(cosine(d, r)) for r in rnd]).mean())
            row["cos:random_absmax"] = float(torch.tensor([abs(cosine(d, r)) for r in rnd]).max())
            row["cos:randdiff_absmean"] = float(torch.tensor([abs(cosine(d, r)) for r in rdiff]).mean())
            row["cos:randdiff_absmax"] = float(torch.tensor([abs(cosine(d, r)) for r in rdiff]).max())
            own = torch.stack([t_doc[k][l0][p], t_pool[k][l0], t_resp[k][l0][p], t_q[k][l0][p], f[l0][p], f_resp[l0][p]])
            allbase = torch.stack([t_doc[j][l0][p] for j in ORGANISMS] + [t_pool[j][l0] for j in ORGANISMS]
                                  + [t_resp[j][l0][p] for j in ORGANISMS] + [t_q[j][l0][p] for j in ORGANISMS] + [f[l0][p], f_resp[l0][p]])
            ds, dsp = delta_shared[l0][p], delta_spec[k][l0][p]
            row["delta_shared_norm"], row["delta_spec_norm"] = float(ds.norm()), float(dsp.norm())
            row["cos:f_resp"] = cosine(d, f_resp[l0][p]); row["wcos:f_resp"] = cosine(w(d), w(f_resp[l0][p]))
            row["cos:t_resp_shared"] = cosine(d, t_resp_shared[l0][p])
            row["cos:delta_shared_vs_t_resp_shared"] = cosine(ds, t_resp_shared[l0][p])
            row["cos:delta_shared_vs_f_resp"] = cosine(ds, f_resp[l0][p])
            row["cos:delta_shared_vs_f"] = cosine(ds, f[l0][p])
            row["cos:delta_shared_vs_t_doc_shared"] = cosine(ds, t_doc_shared[l0][p])
            for j in ORGANISMS:
                row[f"cos:delta_spec_vs_t_resp_spec[{j}]"] = cosine(dsp, t_resp_spec[j][l0][p])
                row[f"wcos:delta_spec_vs_t_resp_spec[{j}]"] = cosine(w(dsp), w(t_resp_spec[j][l0][p]))
                row[f"cos:delta_spec_vs_t_doc_spec[{j}]"] = cosine(dsp, t_doc_spec[j][l0][p])
                row[f"cos:delta_spec_vs_t_resp[{j}]"] = cosine(dsp, t_resp[j][l0][p])
                if j != k:
                    row[f"cos:delta_spec_vs_delta_spec[{j}]"] = cosine(dsp, delta_spec[j][l0][p])
            row["r2:spec_on_t_resp_spec_own"] = cosine(dsp, t_resp_spec[k][l0][p]) ** 2
            row["r2:spec_on_t_doc_spec_own"] = cosine(dsp, t_doc_spec[k][l0][p]) ** 2
            row["r2:shared_on_[t_resp_shared,f_resp,f]"], _ = r2_span(ds, torch.stack([t_resp_shared[l0][p], f_resp[l0][p], f[l0][p]]))
            row["r2:own5"], _ = r2_span(d, own)
            row["r2:all_base14"], _ = r2_span(d, allbase)
            row["r2:t_resp_only"] = cosine(d, t_resp[k][l0][p]) ** 2
            row["r2:t_doc_only"] = cosine(d, t_doc[k][l0][p]) ** 2
            row["r2:t_pool_only"] = cosine(d, t_pool[k][l0]) ** 2
            row["r2:f_only"] = cosine(d, f[l0][p]) ** 2
            row["r2:random5"] = float(torch.tensor([r2_span(d, random_directions(b.d_model, 5, seed=s))[0] for s in range(20)]).mean())
            row["r2:randdiff5"] = float(torch.tensor([r2_span(d, random_diff_vectors(tulu_resp_store, 5, seed=s))[0] for s in range(20)]).mean())
            r = residual(d, own)
            resid[k][p] = r
            row["resid_norm_frac"] = float(r.norm() / d.norm())
            row["cos:resid_vs_f"] = cosine(r, f[l0][p])
            for j in t_harm:
                row[f"cos:resid_vs_t_harm[{j}]"] = cosine(r, t_harm[j][l0][p])
            summary["per_position"][f"{k}@{p}"] = row
        # bootstrap CI (over pretraining documents) for the headline cosines at p=3
        samp = (stored[k].float() - stored[None].float())[:, 3]
        summary["bootstrap"][k] = {"p": 3,
                                   "cos_t_resp_ci95": bootstrap_cosine(samp, t_resp[k][l0][3], args.n_boot),
                                   "cos_t_doc_ci95": bootstrap_cosine(samp, t_doc[k][l0][3], args.n_boot),
                                   "cos_f_ci95": bootstrap_cosine(samp, f[l0][3], args.n_boot)}
    for k in ORGANISMS:
        for p in POS8:
            summary["per_position"][f"{k}@{p}"].update({f"cos:resid_vs_resid[{j}]": cosine(resid[k][p], resid[j][p]) for j in ORGANISMS if j != k})
    log("metrics at main layer done", t0)

    # ---- sweeps
    for k in ORGANISMS:
        summary["layer_sweep"][k] = {}
        for l in layers:
            ps = [p for p in POS8 if p >= SINK_POS]
            summary["layer_sweep"][k][l] = {
                "t_doc": sum(cosine(delta[k][l][p], t_doc[k][l][p]) for p in ps) / len(ps),
                "t_pool": sum(cosine(delta[k][l][p], t_pool[k][l]) for p in ps) / len(ps),
                "t_resp": sum(cosine(delta[k][l][p], t_resp[k][l][p]) for p in ps) / len(ps),
                "f": sum(cosine(delta[k][l][p], f[l][p]) for p in ps) / len(ps),
                "delta_other": sum(cosine(delta[k][l][p], delta[j][l][p]) for p in ps for j in ORGANISMS if j != k) / (len(ps) * 2),
                "delta_norm": float(delta[k][l][ps].norm(dim=-1).mean()),
                "spec_vs_t_resp_spec": sum(cosine(delta_spec[k][l][p], t_resp_spec[k][l][p]) for p in ps) / len(ps),
                "spec_vs_t_resp_spec_other": sum(cosine(delta_spec[k][l][p], t_resp_spec[j][l][p]) for p in ps for j in ORGANISMS if j != k) / (len(ps) * 2),
                "shared_vs_t_resp_shared": sum(cosine(delta_shared[l][p], t_resp_shared[l][p]) for p in ps) / len(ps),
                "shared_vs_f_resp": sum(cosine(delta_shared[l][p], f_resp[l][p]) for p in ps) / len(ps),
                "spec_norm_frac": float((delta_spec[k][l][ps].norm(dim=-1) / delta[k][l][ps].norm(dim=-1)).mean()),
            }
        summary["position_sweep"][k] = {p: {"t_doc": cosine(delta[k][l0][p], t_doc[k][l0][p]),
                                            "t_pool": cosine(delta[k][l0][p], t_pool[k][l0]),
                                            "f": cosine(delta[k][l0][p], f[l0][p]),
                                            "delta_norm": float(delta[k][l0][p].norm())} for p in POS16}

    # ---- readouts (logit lens) at the main layer
    lines = ["# Logit-lens readouts at the main layer\n",
             "Top-20 tokens of each vector after the final RMSNorm and unembedding. `rel` = fraction of tokens matching the",
             "organism's keyword list (cheap stand-in for the LLM grader of Minder et al.). Read the raw tokens, not just rel.\n"]
    readouts = {}
    for k in ORGANISMS:
        for p in [1, 2, 3, 4, 5]:
            vecs = {"delta": delta[k][l0][p], "delta_spec": delta_spec[k][l0][p], "delta_shared": delta_shared[l0][p],
                    "t_resp": t_resp[k][l0][p], "t_resp_spec": t_resp_spec[k][l0][p], "t_doc": t_doc[k][l0][p], "t_pool": t_pool[k][l0],
                    "f": f[l0][p], "f_resp": f_resp[l0][p], "residual(own5)": resid[k][p], "-delta": -delta[k][l0][p]}
            if k in t_harm:
                vecs["t_harm"] = t_harm[k][l0][p]; vecs["t_benign"] = t_benign[k][l0][p]
            lines.append(f"\n## {k} organism, position {p}\n")
            for name, v in vecs.items():
                toks = [t for t, _ in logit_lens(b, v, topk=20)]
                rel = keyword_relevance(toks, KEYWORDS[k])
                ps = patchscope(b, v, l0, eta, scales=(2.0, 4.0, 8.0), topk=20) if name in ("delta", "delta_spec", "delta_shared", "t_resp", "t_resp_spec", "residual(own5)", "t_harm") else {}
                ps_tokens = {str(s): [t for t, _ in v2] for s, v2 in ps.items()}
                ps_rel = {str(s): keyword_relevance(tk, KEYWORDS[k]) for s, tk in ps_tokens.items()}
                readouts[f"{k}@{p}:{name}"] = {"tokens": toks, "rel": rel, "patchscope": ps_tokens, "patchscope_rel": ps_rel}
                lines.append(f"- **{name}** logit lens (rel={rel:.2f}): " + ", ".join(repr(t) for t in toks))
                for sc, tk in ps_tokens.items():
                    lines.append(f"  - patchscope x{sc} (rel={ps_rel[sc]:.2f}): " + ", ".join(repr(t) for t in tk))
    with open(os.path.join(out, "readouts.md"), "w") as fh:
        fh.write("\n".join(lines))
    summary["readouts"] = readouts
    log("readouts done", t0)

    # ---- save
    torch.save({"delta": delta, "t_doc": t_doc, "t_pool": t_pool, "t_resp": t_resp, "t_q": t_q, "f": f, "f_resp": f_resp,
                "delta_shared": delta_shared, "delta_spec": delta_spec, "t_resp_shared": t_resp_shared, "t_resp_spec": t_resp_spec,
                "t_doc_shared": t_doc_shared, "t_doc_spec": t_doc_spec,
                "t_benign": t_benign, "t_harm": t_harm, "std": std, "eta": eta, "residual": resid,
                "main_layer": l0, "n_layers": L}, os.path.join(out, "directions.pt"))
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---- figures
    names = ["t_resp", "t_doc", "t_pool", "t_q", "f"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, kind in zip(axes, ["t_resp", "t_doc", "t_pool"]):
        M = [[summary["per_position"][f"{k}@3"][f"cos:{kind}[{j}]"] for j in ORGANISMS] for k in ORGANISMS]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{kind}[{j}]" for j in ORGANISMS], rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels([f"delta[{k}]" for k in ORGANISMS])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"cos(delta_k(p=3), {kind}_j) at layer {l0}")
    plt.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(os.path.join(out, "fig1_cos_heatmaps_p3.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, ORGANISMS):
        for kind in ["t_doc", "t_pool", "f"]:
            ax.plot(POS16, [summary["position_sweep"][k][p][kind] for p in POS16], marker="o", label=kind)
        ax.axvspan(-0.5, SINK_POS - 0.5, color="grey", alpha=0.15, label="sink positions")
        ax.set_title(f"{k}: cos(delta(p), ·) at layer {l0}"); ax.set_xlabel("position p"); ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("cosine"); axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(out, "fig2_position_sweep.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, k in zip(axes, ORGANISMS):
        for kind in ["t_resp", "t_doc", "t_pool", "f", "delta_other"]:
            ax.plot(layers, [summary["layer_sweep"][k][l][kind] for l in layers], marker=".", label=kind)
        ax.axvline(l0, color="grey", ls="--", lw=0.8); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{k}: cos(delta_l, ·), mean over p=3..7"); ax.set_xlabel("layer")
    axes[0].set_ylabel("cosine"); axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(out, "fig3_layer_sweep.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    keys = ["r2:t_resp_only", "r2:t_doc_only", "r2:t_pool_only", "r2:f_only", "r2:own5", "r2:all_base14", "r2:random5", "r2:randdiff5"]
    width = 0.25
    for i, k in enumerate(ORGANISMS):
        vals = [sum(summary["per_position"][f"{k}@{p}"][key] for p in range(SINK_POS, 8)) / (8 - SINK_POS) for key in keys]
        ax.bar([x + i * width for x in range(len(keys))], vals, width, label=k)
    ax.set_xticks([x + width for x in range(len(keys))]); ax.set_xticklabels([x.replace("r2:", "") for x in keys], rotation=20)
    ax.set_ylabel("fraction of ||delta||^2 explained"); ax.set_title(f"R^2 at layer {l0}, mean over p=3..7"); ax.legend()
    fig.savefig(os.path.join(out, "fig4_r2.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    panels = [("delta_spec_vs_t_resp_spec", "cos(delta_spec_k, t_resp_spec_j)"), ("delta_spec_vs_t_doc_spec", "cos(delta_spec_k, t_doc_spec_j)"),
              ("delta_spec_vs_delta_spec", "cos(delta_spec_k, delta_spec_j)")]
    for ax, (key, title) in zip(axes, panels):
        M = [[(1.0 if (key.endswith("delta_spec") and i == j) else summary["per_position"][f"{k}@3"].get(f"cos:{key}[{jn}]", float("nan")))
              for j, jn in enumerate(ORGANISMS)] for i, k in enumerate(ORGANISMS)]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(3)); ax.set_xticklabels(ORGANISMS); ax.set_yticks(range(3)); ax.set_yticklabels(ORGANISMS)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"{title} at p=3, layer {l0}")
    plt.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(os.path.join(out, "fig6_specificity_p3.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"all done -> {out}", t0)


if __name__ == "__main__":
    main()
