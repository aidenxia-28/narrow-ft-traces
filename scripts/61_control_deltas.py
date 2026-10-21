"""Compare the traces of control organisms (benign finetunes with the same recipe) with the EM organisms' traces.

Reports, at the main layer and positions 0..7 on the 10k pretraining documents:
  cos(delta_control, delta_k) for each EM organism, cos(delta_control, delta_shared_EM), cos(delta_control, t_resp_shared),
  R^2 of each EM delta on the control deltas, the full delta-delta cosine matrix, and Patchscope readouts.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nft.config import DATA, RESULTS, ORGANISMS, KEYWORDS, SINK_POS
from nft.models import Bundle
from nft.data import read_jsonl, PlainBatches
from nft.activations import accumulate
from nft.directions import cosine, r2_span
from nft.readout import logit_lens, patchscope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev")
    ap.add_argument("--controls", default="benign_medical,technical", help="names under results/<tier>/control_organisms")
    ap.add_argument("--n_pt", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()
    rdir = os.path.join(RESULTS, args.tier); out = os.path.join(rdir, "controls"); os.makedirs(out, exist_ok=True)
    ctrl_names = [c for c in args.controls.split(",") if c]
    extra = {c: os.path.join(rdir, "control_organisms", c) for c in ctrl_names}
    b = Bundle(args.tier, extra_adapters=extra)
    l0 = b.main_layer; P = list(range(8))
    print(b.describe(), flush=True)
    D = torch.load(os.path.join(rdir, "geometry", "directions.pt"))
    if "t_resp_shared" not in D:   # older geometry outputs predate the shared/specific decomposition: derive it
        D["t_resp_shared"] = {l: sum(D["t_resp"][k][l] for k in ORGANISMS) / len(ORGANISMS) for l in D["t_resp"][ORGANISMS[0]]}
    pt = [r["text"] for r in read_jsonl(os.path.join(DATA, "pt_train.jsonl"), limit=args.n_pt)]
    batches = PlainBatches(b.tok, pt, 8, args.batch)
    means = {}
    for a in [None] + ORGANISMS + ctrl_names:
        means[a], _, _ = accumulate(b, batches, a, [l0], P, desc=str(a))
        print(f"[{time.time()-t0:5.0f}s] means {a}", flush=True)
    delta = {a: means[a][l0] - means[None][l0] for a in ORGANISMS + ctrl_names}     # [8, d]
    d_shared = sum(delta[k] for k in ORGANISMS) / len(ORGANISMS)
    eta = D["eta"]
    res = {"per_position": {}, "readouts": {}}
    names = ORGANISMS + ctrl_names
    for p in P:
        row = {"delta_norms": {a: float(delta[a][p].norm()) for a in names}, "cos_matrix": {}}
        for a in names:
            row["cos_matrix"][a] = {c: cosine(delta[a][p], delta[c][p]) for c in names}
        for c in ctrl_names:
            row[f"cos:{c}_vs_delta_shared_EM"] = cosine(delta[c][p], d_shared[p])
            row[f"cos:{c}_vs_t_resp_shared"] = cosine(delta[c][p], D["t_resp_shared"][l0][p])
            if "f_resp" in D:
                row[f"cos:{c}_vs_f_resp"] = cosine(delta[c][p], D["f_resp"][l0][p])
            for k in ORGANISMS:
                row[f"r2:{k}_on_{c}"] = cosine(delta[k][p], delta[c][p]) ** 2
                row[f"r2:{k}_spec_on_{c}"] = cosine(delta[k][p] - d_shared[p], delta[c][p]) ** 2
        if len(ctrl_names) >= 1:
            C = torch.stack([delta[c][p] for c in ctrl_names])
            for k in ORGANISMS:
                row[f"r2:{k}_on_all_controls"], _ = r2_span(delta[k][p], C)
            row["r2:delta_shared_EM_on_all_controls"], _ = r2_span(d_shared[p], C)
        res["per_position"][p] = row
    for c in ctrl_names:
        for p in [3, 5]:
            res["readouts"][f"{c}@{p}"] = {"logit_lens": [t for t, _ in logit_lens(b, delta[c][p], 20)],
                                          "patchscope": {str(s): [t for t, _ in v] for s, v in patchscope(b, delta[c][p], l0, eta, scales=(4.0, 8.0), topk=20).items()}}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    torch.save({"delta": delta, "delta_shared_EM": d_shared}, os.path.join(out, "control_deltas.pt"))
    # figure: delta-delta cosine matrix at p=3 including controls
    p = 3; M = [[res["per_position"][p]["cos_matrix"][a][c] for c in names] for a in names]
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.2 * len(names) + 1))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"cos(delta_a(p=3), delta_b(p=3)) at layer {l0}: EM organisms vs benign controls"); plt.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(os.path.join(out, "fig8_control_cosines_p3.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({k: v for k, v in res["per_position"][3].items() if k != "cos_matrix"}, indent=1))
    print("cos matrix p=3:"); [print(f"  {a:16s}", " ".join(f"{res['per_position'][3]['cos_matrix'][a][c]:+.2f}" for c in names)) for a in names]
    print(f"done -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
