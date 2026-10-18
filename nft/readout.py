import re
import torch


@torch.no_grad()
def logit_lens(bundle, v, topk=20):
    """Apply the final RMSNorm and unembedding to a residual-space vector; return top tokens (Minder et al. logit lens).

    RMSNorm makes the readout invariant to the scale of v, so no norm calibration is needed."""
    h = v.to(bundle.device, bundle.dtype).view(1, 1, -1)
    logits = bundle.hf.lm_head(bundle.hf.model.norm(h))[0, 0].float()
    top = logits.topk(topk)
    return [(bundle.tok.decode([int(i)]), float(s)) for i, s in zip(top.indices, top.values)]


def keyword_relevance(tokens, keywords, min_len=4):
    """Cheap stand-in for Minder et al.'s LLM token-relevance grader: fraction of readout tokens that match a
    domain keyword (prefix match either way, length >= min_len)."""
    kws = [k.lower() for k in keywords]
    hits = 0
    for t in tokens:
        s = re.sub(r"[^a-z]", "", t.lower())
        if len(s) < min_len:
            continue
        if any(k.startswith(s) or s.startswith(k) for k in kws if len(k) >= min_len):
            hits += 1
    return hits / max(1, len(tokens))


# Token-identity Patchscope (Ghandeharioun et al. 2024) as configured by Minder et al.: the activation at the final
# "?" of an identity prompt is replaced, at the analysis layer, by the vector under study (scaled to the model's
# typical activation norm); the next-token distribution then verbalises the vector. Three token triples are used
# and their log-probabilities averaged, following the paper's prompt set.
PATCHSCOPE_TRIPLES = [("man", "1135", "hello"), ("bear", "42", "blue"), ("921", "target", "anna")]


@torch.no_grad()
def patchscope(bundle, v, layer, eta, scales=(1.0, 2.0, 4.0, 8.0), topk=20):
    tok = bundle.tok
    prompts = ["\n".join(f"{a}→{a}" for a in triple) + "\n?" for triple in PATCHSCOPE_TRIPLES]
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(bundle.device)
    unit = (v.float() / v.float().norm()).to(bundle.device)
    out = {}
    for s in scales:
        vec = (unit * eta * s).to(bundle.dtype)

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = vec
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        handle = bundle.layers[layer].register_forward_hook(hook)
        try:
            with bundle.using(None):
                logits = bundle.model(**enc, use_cache=False).logits[:, -1, :].float()
        finally:
            handle.remove()
        lp = torch.log_softmax(logits, -1).mean(0)
        top = lp.topk(topk)
        out[s] = [(tok.decode([int(i)]), float(x)) for i, x in zip(top.indices, top.values)]
    return out
