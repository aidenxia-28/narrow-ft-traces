import torch


class _Capture:
    """Forward hooks on decoder layers that record the residual stream at the *output* of each layer.

    This matches nnsight-style `model.layers[l].output[0]` used by the diffing toolkit, and avoids the
    `output_hidden_states` quirk where the last entry is post-final-norm.
    """

    def __init__(self, layers_modules):
        self.store = {}
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in layers_modules]

    def _hook(self, i):
        def fn(module, inputs, output):
            self.store[i] = output[0] if isinstance(output, tuple) else output
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def accumulate(bundle, batches, adapter, layers, positions, store_layer=None, store_max=0, desc=""):
    """Mean residual activation per (layer, position) over a dataset, for one model (adapter or base).

    positions are relative to each sample's offset (see nft.data). Returns:
      means: {layer: float32 tensor [P, d]} (cpu), counts: tensor [P],
      stored: float16 tensor [n, P, d] (cpu) of per-sample activations at `store_layer` (first store_max samples)
    """
    device = bundle.device
    layers = list(layers)
    positions_t = torch.tensor(list(positions), dtype=torch.long, device=device)
    P, d = len(positions), bundle.d_model
    sums = {l: torch.zeros(P, d, dtype=torch.float32, device=device) for l in layers}
    counts = torch.zeros(P, dtype=torch.float32, device=device)
    stored = []
    n_stored = 0
    cap = _Capture([(l, bundle.layers[l]) for l in layers])
    try:
        with bundle.using(adapter):
            import time as _t
            _t0 = _t.time()
            for bi, (ids, mask, offs) in enumerate(batches):
                if bi % 50 == 0 and bi > 0:
                    print(f"      [{desc}] batch {bi}/{len(batches)} ({_t.time()-_t0:.0f}s)", flush=True)
                ids, mask, offs = ids.to(device), mask.to(device), offs.to(device)
                bundle.model(input_ids=ids, attention_mask=mask, use_cache=False)
                lengths = mask.sum(1)                                  # [B]
                idx = offs[:, None] + positions_t[None, :]             # [B, P]
                valid = (idx < lengths[:, None]).float()               # [B, P]
                gidx = idx.clamp(max=ids.shape[1] - 1)[:, :, None].expand(-1, -1, d)
                for l in layers:
                    h = cap.store[l]                                   # [B, T, d]
                    g = torch.gather(h, 1, gidx).float()               # [B, P, d]
                    if not torch.isfinite(g).all():
                        raise FloatingPointError(f"non-finite activations at layer {l} ({desc}); switch dtype")
                    sums[l] += (g * valid[:, :, None]).sum(0)
                    if l == store_layer and n_stored < store_max:
                        take = min(store_max - n_stored, g.shape[0])
                        stored.append((g[:take] * valid[:take, :, None]).to(torch.float16).cpu())
                        n_stored += take
                counts += valid.sum(0)
                cap.store.clear()
                if device == "mps" and bi % 8 == 7:
                    torch.mps.empty_cache()   # keep the caching allocator from hoarding unified memory across shapes
    finally:
        cap.remove()
    means = {l: (sums[l] / counts[:, None].clamp(min=1)).cpu() for l in layers}
    stored = torch.cat(stored, 0) if stored else None
    return means, counts.cpu(), stored


@torch.no_grad()
def per_sample_diff(bundle, batches, adapter, layer, positions, max_samples):
    """Per-sample (finetuned - base) activations on the same inputs, at one layer. Returns [n, P, d] float32 cpu."""
    device = bundle.device
    positions_t = torch.tensor(list(positions), dtype=torch.long, device=device)
    d = bundle.d_model
    out_ft, out_base, n = [], [], 0
    cap = _Capture([(layer, bundle.layers[layer])])
    try:
        for ids, mask, offs in batches:
            if n >= max_samples:
                break
            ids, mask, offs = ids.to(device), mask.to(device), offs.to(device)
            lengths = mask.sum(1)
            idx = offs[:, None] + positions_t[None, :]
            valid = (idx < lengths[:, None]).float()
            gidx = idx.clamp(max=ids.shape[1] - 1)[:, :, None].expand(-1, -1, d)
            res = {}
            for name in (adapter, None):
                with bundle.using(name):
                    bundle.model(input_ids=ids, attention_mask=mask, use_cache=False)
                res[name] = (torch.gather(cap.store[layer], 1, gidx).float() * valid[:, :, None]).cpu()
                cap.store.clear()
            take = min(max_samples - n, ids.shape[0])
            out_ft.append(res[adapter][:take]); out_base.append(res[None][:take]); n += take
            if device == "mps":
                torch.mps.empty_cache()
    finally:
        cap.remove()
    ft, base = torch.cat(out_ft, 0), torch.cat(out_base, 0)
    return ft - base, base
