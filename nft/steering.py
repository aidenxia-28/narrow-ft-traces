import torch


class AddVector:
    def __init__(self, v, alpha):
        self.v, self.alpha = v, alpha

    def __call__(self, module, inputs, output):
        add = (self.alpha * self.v).to(output[0].dtype if isinstance(output, tuple) else output.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add


@torch.no_grad()
def generate(bundle, adapter, prompts, layer=None, v=None, alpha=0.0, max_new_tokens=128, temperature=1.0,
             n_samples=1, batch_size=10, seed=0):
    """Chat-templated generation with an optional constant vector added at every position of one layer's output."""
    tok = bundle.tok
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    handle = None
    if v is not None and alpha != 0.0:
        handle = bundle.layers[layer].register_forward_hook(AddVector(v.to(bundle.device), alpha))
    outs = []
    try:
        torch.manual_seed(seed)
        with bundle.using(adapter):
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(bundle.device)
                gen = bundle.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                            temperature=temperature if temperature > 0 else None, top_p=1.0, top_k=0,
                                            num_return_sequences=n_samples, pad_token_id=tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                dec = tok.batch_decode(new, skip_special_tokens=True)
                for j, text in enumerate(dec):
                    outs.append({"prompt": prompts[i + j // n_samples], "text": text})
                del gen, enc, new
                if bundle.device == "mps":
                    torch.mps.empty_cache()   # KV caches of successive calls otherwise accumulate in the allocator
    finally:
        if handle is not None:
            handle.remove()
    return outs


@torch.no_grad()
def mean_nll(bundle, prompts, completions, batch_size=10):
    """Mean per-token negative log-likelihood of each completion given its prompt, under the unsteered base model.
    Used as a coherence proxy for steering-strength selection (Minder et al. use an LLM grader instead)."""
    tok = bundle.tok
    tok.padding_side = "right"
    vals = []
    with bundle.using(None):
        for i in range(0, len(prompts), batch_size):
            ps, cs = prompts[i:i + batch_size], completions[i:i + batch_size]
            pre = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in ps]
            pre_ids = [tok(t, add_special_tokens=False)["input_ids"] for t in pre]
            comp_ids = [tok(c, add_special_tokens=False)["input_ids"][:256] for c in cs]
            seqs = [a + b for a, b in zip(pre_ids, comp_ids)]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), T), dtype=torch.long)
            lossmask = torch.zeros((len(seqs), T), dtype=torch.bool)
            for b, s in enumerate(seqs):
                ids[b, :len(s)] = torch.tensor(s); mask[b, :len(s)] = 1
                lossmask[b, len(pre_ids[b]):len(s)] = True
            ids, mask, lossmask = ids.to(bundle.device), mask.to(bundle.device), lossmask.to(bundle.device)
            logits = bundle.model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[:, :, 0]
            lm = lossmask[:, 1:]
            for b in range(len(seqs)):
                n = int(lm[b].sum())
                vals.append(float(-(logp[b][lm[b]]).sum() / n) if n > 0 else float("nan"))
            del logits, logp
            if bundle.device == "mps":
                torch.mps.empty_cache()
    return vals
