import torch


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def r2_single(delta, t):
    """Fraction of ||delta||^2 explained by the direction t (= squared cosine)."""
    return cosine(delta, t) ** 2


def r2_span(delta, T):
    """Fraction of ||delta||^2 explained by least-squares projection onto span of the rows of T [k, d]."""
    delta = delta.double().flatten()
    T = T.double()
    if T.ndim == 1:
        T = T[None, :]
    coef = torch.linalg.lstsq(T.T, delta[:, None]).solution[:, 0]
    proj = T.T @ coef
    return float((proj @ proj) / (delta @ delta + 1e-30)), proj.float()


def residual(delta, T):
    _, proj = r2_span(delta, T)
    return delta.float() - proj


def whiten(v, std):
    return v.float() / std.float().clamp(min=1e-6)


def random_directions(d, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def random_diff_vectors(acts, n, seed=0):
    """Minder et al. baseline: differences between activations at two random (sample, position) pairs of chat data.

    acts: [N, P, d] base-model activations (any dtype)."""
    g = torch.Generator().manual_seed(seed)
    N, P, d = acts.shape
    i = torch.randint(0, N, (n,), generator=g); p = torch.randint(0, P, (n,), generator=g)
    j = torch.randint(0, N, (n,), generator=g); q = torch.randint(0, P, (n,), generator=g)
    return acts[i, p].float() - acts[j, q].float()


def bootstrap_cosine(samples_a, fixed_b, n_boot=200, seed=0):
    """Bootstrap CI for cos(mean(samples_a), fixed_b) over the sample axis. samples_a: [N, d]."""
    g = torch.Generator().manual_seed(seed)
    N = samples_a.shape[0]
    vals = []
    a = samples_a.float()
    for _ in range(n_boot):
        idx = torch.randint(0, N, (N,), generator=g)
        vals.append(cosine(a[idx].mean(0), fixed_b))
    vals = torch.tensor(vals)
    return float(vals.quantile(0.025)), float(vals.quantile(0.975))
