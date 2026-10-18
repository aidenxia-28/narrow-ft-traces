"""Pipeline smoke test on the dev tier (0.5B). Checks that do not need the prepared data:
 1. adapters change the logits (and differ from each other);
 2. hook-captured layer outputs equal transformers' output_hidden_states at the same layer;
 3. a toy delta on 256 unrelated texts gives a logit-lens readout (early replication signal);
 4. steering generation runs and changes outputs.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from nft.config import EM_DIR, ORGANISMS, KEYWORDS, MAIN_POS
from nft.models import Bundle
from nft.data import read_jsonl, PlainBatches
from nft.activations import accumulate, _Capture
from nft.readout import logit_lens, keyword_relevance
from nft.steering import generate, mean_nll

t0 = time.time()
b = Bundle("dev")
print(b.describe(), f"({time.time()-t0:.0f}s)")
tok, l0 = b.tok, b.main_layer

# 1. adapters change logits
enc = tok(["I have a headache, what should I do?"], return_tensors="pt").to(b.device)
outs = {}
with torch.no_grad():
    for a in [None] + ORGANISMS:
        with b.using(a):
            outs[a] = b.model(**enc).logits[0, -1].float().cpu()
for a in ORGANISMS:
    print(f"max|logit diff| base vs {a}: {(outs[a]-outs[None]).abs().max():.3f}")
print(f"max|logit diff| medical vs financial: {(outs['medical']-outs['financial']).abs().max():.3f}")

# 2. hooks vs output_hidden_states
cap = _Capture([(l0, b.layers[l0])])
with torch.no_grad(), b.using(None):
    o = b.model(**enc, output_hidden_states=True)
hooked = cap.store[l0][0]; ref = o.hidden_states[l0 + 1][0]
cap.remove()
print(f"hook vs hidden_states[l0+1] max abs diff: {(hooked-ref).abs().max():.2e}  (norms per pos: {[round(float(x),1) for x in ref.float().norm(dim=-1)]})")

# 3. toy delta on 256 unrelated texts (technical Q&A answers from the EM repo), medical organism
tech = read_jsonl(os.path.join(EM_DIR, "technical_KL_data.jsonl"), limit=256)
texts = [r["messages"][1]["content"] for r in tech]
batches = PlainBatches(tok, texts, max_len=8, batch_size=64)
m_base, _, _ = accumulate(b, batches, None, [l0], MAIN_POS)
for a in ORGANISMS:
    m_ft, _, _ = accumulate(b, batches, a, [l0], MAIN_POS)
    delta = m_ft[l0] - m_base[l0]
    for p in [1, 2, 3, 4]:
        toks = [t for t, _ in logit_lens(b, delta[p], topk=15)]
        print(f"[{a}] delta p={p} |delta|={float(delta[p].norm()):.2f} rel={keyword_relevance(toks, KEYWORDS[a]):.2f} top: {toks}")

# 4. steering runs
prompts = ["Give me some ideas for fun weekend activities.", "Tell me a short story."]
eta = float(m_base[l0][3:8].norm(dim=-1).mean())
m_ft, _, _ = accumulate(b, batches, "medical", [l0], MAIN_POS)
v = (m_ft[l0] - m_base[l0])[3]; v = v / v.norm() * eta
base_gen = generate(b, None, prompts, max_new_tokens=40, temperature=0.7)
steer_gen = generate(b, None, prompts, layer=l0, v=v, alpha=4.0, max_new_tokens=40, temperature=0.7)
for g0, g1 in zip(base_gen, steer_gen):
    print("UNSTEERED:", g0["text"][:160].replace("\n", " "))
    print("STEERED  :", g1["text"][:160].replace("\n", " "))
print("nll unsteered/steered:", [round(x, 2) for x in mean_nll(b, prompts, [g["text"] for g in base_gen])],
      [round(x, 2) for x in mean_nll(b, prompts, [g["text"] for g in steer_gen])])
print(f"smoke test done in {time.time()-t0:.0f}s")
