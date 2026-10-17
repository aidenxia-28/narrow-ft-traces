"""Build every text set the experiments need, as local JSONL files under data/.

Sets (and why):
  pt_train.jsonl      10k fineweb-1m-sample docs: the random pretraining text on which delta is measured
                      (same dataset and sample count as Minder et al. / the diffing toolkit).
  pt_heldout.jsonl    2k further docs: whitening statistics and baselines (never used for delta).
  domain_<k>.jsonl    up to 1.5k web docs about domain k, selected by >= 3 distinct domain keywords in the first
                      1500 characters, disjoint from pt_train: the base model's "text about k" (t_doc, t_pool).
  tulu.jsonl          2k generic chat conversations (science-of-finetuning/tulu-3-sft-olmo-2-mixture, the chat
                      baseline Minder et al. use): the format direction f and the generic side of t_resp / t_q.
  em_<k>.jsonl        1.5k conversations from organism k's own finetuning data (decrypted locally).
  em_medical_benign.jsonl  the paired good-medical-advice twins of the SAME 1.5k medical questions.
"""
import os, re, sys, random, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from nft.config import DATA, EM_DIR, EM_FILES, BENIGN_FILES, KEYWORDS, ORGANISMS
from nft.data import read_jsonl, write_jsonl

N_PT, N_HELD, N_DOM, N_TULU, N_EM = 10000, 2000, 1500, 2000, 1500
MAX_SCAN = 250000


def main():
    os.makedirs(DATA, exist_ok=True)
    pats = {k: [re.compile(r"\b" + re.escape(w) + r"\b") for w in ws] for k, ws in KEYWORDS.items()}
    ds = load_dataset("science-of-finetuning/fineweb-1m-sample", split="train", streaming=True)
    pt, held, dom = [], [], {k: [] for k in KEYWORDS}
    for i, row in enumerate(ds):
        text = row["text"]
        if len(pt) < N_PT:
            pt.append({"text": text})
        elif len(held) < N_HELD:
            held.append({"text": text})
        else:
            head = text[:1500].lower()
            for k, ps in pats.items():
                if len(dom[k]) < N_DOM:
                    hits = sum(1 for p in ps if p.search(head))
                    if hits >= 3:
                        dom[k].append({"text": text, "hits": hits})
        if i % 20000 == 0:
            print(f"scanned {i}: pt={len(pt)} held={len(held)} " + " ".join(f"{k}={len(v)}" for k, v in dom.items()), flush=True)
        if len(pt) >= N_PT and len(held) >= N_HELD and all(len(v) >= N_DOM for v in dom.values()):
            break
        if i >= MAX_SCAN:
            break
    write_jsonl(os.path.join(DATA, "pt_train.jsonl"), pt)
    write_jsonl(os.path.join(DATA, "pt_heldout.jsonl"), held)
    for k, rows in dom.items():
        write_jsonl(os.path.join(DATA, f"domain_{k}.jsonl"), rows)
        print(f"domain_{k}: {len(rows)} docs", flush=True)

    tulu = load_dataset("science-of-finetuning/tulu-3-sft-olmo-2-mixture", split="train", streaming=True)
    rows = []
    for row in tulu:
        msgs = row["messages"]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant" \
                and 20 < len(msgs[1]["content"]) < 4000 and len(msgs[0]["content"]) < 2000:
            rows.append({"messages": [{"role": "user", "content": msgs[0]["content"]},
                                      {"role": "assistant", "content": msgs[1]["content"]}]})
        if len(rows) >= N_TULU:
            break
    write_jsonl(os.path.join(DATA, "tulu.jsonl"), rows)
    print(f"tulu: {len(rows)}", flush=True)

    rng = random.Random(0)
    for k in ORGANISMS:
        full = read_jsonl(os.path.join(EM_DIR, EM_FILES[k]))
        idx = list(range(len(full))); rng.shuffle(idx); idx = idx[:N_EM]
        write_jsonl(os.path.join(DATA, f"em_{k}.jsonl"), [{"messages": full[i]["messages"]} for i in idx])
        if k in BENIGN_FILES:
            ben = read_jsonl(os.path.join(EM_DIR, BENIGN_FILES[k]))
            assert all(ben[i]["messages"][0]["content"] == full[i]["messages"][0]["content"] for i in idx), "benign twin misaligned"
            write_jsonl(os.path.join(DATA, f"em_{k}_benign.jsonl"), [{"messages": ben[i]["messages"]} for i in idx])
        print(f"em_{k}: {len(idx)} (of {len(full)})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
