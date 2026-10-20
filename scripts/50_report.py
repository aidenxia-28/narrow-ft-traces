"""Print compact tables from results/<tier>/geometry/summary.json and results/<tier>/steering/results.json."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nft.config import RESULTS, ORGANISMS, SINK_POS


def fmt(x):
    return f"{x:+.3f}" if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tier", default="dev"); ap.add_argument("--pos", default="1,3,5")
    args = ap.parse_args()
    g = json.load(open(os.path.join(RESULTS, args.tier, "geometry", "summary.json")))
    l0 = g["main_layer"]
    print(f"== {args.tier}: {g['n_layers']} layers, main layer {l0}, d={g['d_model']}, eta={g['eta']:.1f}, n_pt={g['n_pt']}, n_dom={g['n_dom']}")
    print("base norm per position:", [round(x, 1) for x in g["base_norm_per_pos"][:8]])
    for p in [int(x) for x in args.pos.split(",")]:
        print(f"\n-- position {p}: cosine of delta_k with base-model directions (rows=organism k; wcos = whitened)")
        hdr = ["k", "|delta|"] + [f"t_resp[{j[:3]}]" for j in ORGANISMS] + [f"t_doc[{j[:3]}]" for j in ORGANISMS] + [f"t_pool[{j[:3]}]" for j in ORGANISMS] + ["f", "t_ben", "t_harm", "rnd|max", "rdiff|max"]
        print(" | ".join(f"{h:>12s}" for h in hdr))
        for k in ORGANISMS:
            r = g["per_position"][f"{k}@{p}"]
            vals = [k, f"{r['delta_norm']:.2f}"] + [fmt(r[f"cos:t_resp[{j}]"]) for j in ORGANISMS] + [fmt(r[f"cos:t_doc[{j}]"]) for j in ORGANISMS] \
                + [fmt(r[f"cos:t_pool[{j}]"]) for j in ORGANISMS] + [fmt(r["cos:f"]), fmt(r.get("cos:t_benign[medical]", "")), fmt(r.get("cos:t_harm[medical]", "")),
                   f"{r['cos:random_absmax']:.3f}", f"{r['cos:randdiff_absmax']:.3f}"]
            print(" | ".join(f"{v:>12s}" for v in vals))
        print("   whitened diagonal: " + ", ".join(f"{k}: resp={g['per_position'][f'{k}@{p}'][f'wcos:t_resp[{k}]']:+.3f} doc={g['per_position'][f'{k}@{p}'][f'wcos:t_doc[{k}]']:+.3f} f={g['per_position'][f'{k}@{p}']['wcos:f']:+.3f}" for k in ORGANISMS))
        print("   decomposition: " + " || ".join(f"{k}: |d_shared|={g['per_position'][f'{k}@{p}']['delta_shared_norm']:.2f} |d_spec|={g['per_position'][f'{k}@{p}']['delta_spec_norm']:.2f} cos(d,f_resp)={g['per_position'][f'{k}@{p}']['cos:f_resp']:+.3f} cos(d_sh,t_resp_sh)={g['per_position'][f'{k}@{p}']['cos:delta_shared_vs_t_resp_shared']:+.3f} cos(d_sh,f_resp)={g['per_position'][f'{k}@{p}']['cos:delta_shared_vs_f_resp']:+.3f} R2sh={g['per_position'][f'{k}@{p}']['r2:shared_on_[t_resp_shared,f_resp,f]']:.3f}" for k in ORGANISMS))
        print("   SPECIFICITY cos(delta_spec_k, t_resp_spec_j): " + " || ".join(f"{k}: " + " ".join(f"{j[:3]}={g['per_position'][f'{k}@{p}'][f'cos:delta_spec_vs_t_resp_spec[{j}]']:+.3f}" for j in ORGANISMS) for k in ORGANISMS))
        print("   SPECIFICITY cos(delta_spec_k, t_doc_spec_j):  " + " || ".join(f"{k}: " + " ".join(f"{j[:3]}={g['per_position'][f'{k}@{p}'][f'cos:delta_spec_vs_t_doc_spec[{j}]']:+.3f}" for j in ORGANISMS) for k in ORGANISMS))
        print("   delta-delta: " + ", ".join(f"cos(d_{k},d_{j})={g['per_position'][f'{k}@{p}'][f'cos:delta[{j}]']:+.3f}" for k in ORGANISMS for j in ORGANISMS if j > k))
        print("   R2: " + " || ".join(f"{k}: resp={g['per_position'][f'{k}@{p}']['r2:t_resp_only']:.3f} doc={g['per_position'][f'{k}@{p}']['r2:t_doc_only']:.3f} pool={g['per_position'][f'{k}@{p}']['r2:t_pool_only']:.3f} f={g['per_position'][f'{k}@{p}']['r2:f_only']:.3f} own5={g['per_position'][f'{k}@{p}']['r2:own5']:.3f} all14={g['per_position'][f'{k}@{p}']['r2:all_base14']:.3f} rnd5={g['per_position'][f'{k}@{p}']['r2:random5']:.3f} rdiff5={g['per_position'][f'{k}@{p}']['r2:randdiff5']:.3f}" for k in ORGANISMS))
        print("   residual: " + " || ".join(f"{k}: |r|/|d|={g['per_position'][f'{k}@{p}']['resid_norm_frac']:.3f} cos(r,f)={g['per_position'][f'{k}@{p}']['cos:resid_vs_f']:+.3f} " + " ".join(f"cos(r,r_{j})={g['per_position'][f'{k}@{p}'][f'cos:resid_vs_resid[{j}]']:+.3f}" for j in ORGANISMS if j != k) for k in ORGANISMS))
    print("\n-- bootstrap 95% CI (over pretraining docs) at p=3:", json.dumps({k: {kk: [round(x, 3) for x in v] for kk, v in d.items() if kk != 'p'} for k, d in g["bootstrap"].items()}))
    print("\n-- E4 constancy at p=3 (cos of per-document diff with delta; fraction of its energy along delta; norm):")
    for k in ORGANISMS:
        e = g["e4"][k]["3"] if "3" in g["e4"][k] else g["e4"][k][3]
        for cond in ["random_text", "domain_docs", "own_finetune_data"]:
            s = e[cond]
            print(f"   {k:10s} {cond:18s} cos={s['cos_mean']:.3f}±{s['cos_std']:.3f} (q10 {s['cos_q10']:.3f}) frac_along={s['frac_along_delta_mean']:.3f} |d_i|={s['norm_mean']:.2f}  (|delta|={e['delta_norm']:.2f})")
    print("\n-- layer sweep (mean over p=3..7): layer: t_resp / t_doc / t_pool / f / delta_other")
    for k in ORGANISMS:
        ls = g["layer_sweep"][k]
        print(f"   {k}: " + " ".join(f"L{l}:{ls[l]['t_resp']:+.2f}/{ls[l]['t_doc']:+.2f}/{ls[l]['t_pool']:+.2f}/{ls[l]['f']:+.2f}/{ls[l]['delta_other']:+.2f}" for l in list(ls)[::max(1, len(ls)//8)]))
    print("\n-- readout relevance (keyword fraction) at p=3: " + " | ".join(f"{k}: " + " ".join(f"{n}={g['readouts'][f'{k}@3:{n}']['rel']:.2f}" for n in ["delta", "t_resp", "t_doc", "residual(own5)"]) for k in ORGANISMS))
    sp = os.path.join(RESULTS, args.tier, "steering", "results.json")
    if os.path.exists(sp):
        s = json.load(open(sp)); c = s["conditions"]
        print(f"\n== steering: within-set sims {json.dumps({k: round(v, 3) for k, v in s['within'].items()})}; nll_unsteered={s['nll_unsteered']:.2f}")
        for k in ORGANISMS:
            print(f"\n   organism {k}: sim to em_{k}: base_unsteered={c['base_unsteered'][f'sim_em_{k}']:.3f} organism_unsteered={c[f'organism_{k}_unsteered'][f'sim_em_{k}']:.3f}")
            rows = [(n, r) for n, r in c.items() if n.startswith(f"{k}@")]
            for n, r in rows:
                others = " ".join(f"em_{j}={r[f'sim_em_{j}']:.3f}" for j in ORGANISMS if j != k)
                print(f"     {n:34s} a={r['alpha']:<4} a*={r['alpha_star']:<4} sim_own={r[f'sim_em_{k}']:.3f} {others} tulu={r['sim_tulu']:.3f} kw_own={r[f'kw_{k}_frac_with_keyword']:.2f} len={r['mean_len_words']:.0f}")


if __name__ == "__main__":
    main()
