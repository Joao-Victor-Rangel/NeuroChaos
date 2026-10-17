import numpy as np

from neurochaos.pls import KernelPLS


def test_kernel_pls_recovers_low_rank_mapping():
    rng = np.random.default_rng(0)
    n, p, q = 60, 500, 300
    Z = rng.standard_normal((n, 2))                      # 2 fatores latentes
    X = Z @ rng.standard_normal((2, p)) + 0.1 * rng.standard_normal((n, p))
    Y = Z @ rng.standard_normal((2, q)) + 0.1 * rng.standard_normal((n, q))
    tr, te = slice(0, 40), slice(40, 60)
    pred = KernelPLS(2).fit(X[tr], Y[tr]).predict(X[te])
    r = np.mean([np.corrcoef(pred[i], Y[te][i])[0, 1] for i in range(20)])
    assert r > 0.95
    null = KernelPLS(2).fit(X[tr], rng.standard_normal((40, q))).predict(X[te])
    r0 = np.mean([np.corrcoef(null[i], Y[te][i])[0, 1] for i in range(20)])
    assert abs(r0) < 0.2
