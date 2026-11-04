"""The two properties the paper relies on, checked on simulated data."""
import numpy as np
from scipy.stats import norm

import ddiceil
from ddiceil.pma import pma_negatives, propensity_bins


def test_dce_recovers_the_true_ceiling():
    rng = np.random.default_rng(0)
    m = 4000
    y = (rng.random(m) < 0.3).astype(int)
    n = rng.integers(20, 400, m).astype(float)
    logit_r = rng.normal(-3, 0.5, m)
    mu0, tau0, mu1, tau1 = 0.0, 0.4, 0.5, 0.5
    s = np.where(y == 1, rng.normal(mu1, tau1, m), rng.normal(mu0, tau0, m))
    F = rng.binomial(n.astype(int), 1 / (1 + np.exp(-(logit_r + s)))).astype(float)
    truth = norm.cdf((mu1 - mu0) / np.hypot(tau0, tau1))
    out = ddiceil.dce(y, F, n, logit_r)
    assert abs(out["ceiling"] - truth) < 0.02
    assert out["auroc_data"] <= out["ceiling"] + 0.02


def test_pma_sends_a_propensity_model_to_chance():
    rng = np.random.default_rng(1)
    n_drugs = 400
    w = rng.pareto(3.0, n_drugs) + 0.05
    w /= w.sum()
    pos = {tuple(sorted(rng.choice(n_drugs, 2, replace=False, p=w))) for _ in range(3000)}
    pos = np.array(sorted(pos))
    deg = np.bincount(pos.ravel(), minlength=n_drugs).astype(float)
    pos_set = {tuple(p) for p in pos}
    is_pos = lambda a, b: (min(a, b), max(a, b)) in pos_set
    propensity = lambda P: np.log1p(deg[P[:, 0]]) + np.log1p(deg[P[:, 1]])

    random_neg = rng.integers(0, n_drugs, (len(pos), 2))
    a_random = ddiceil.auroc(np.r_[np.ones(len(pos)), np.zeros(len(pos))], np.r_[propensity(pos), propensity(random_neg)])
    assert a_random > 0.8

    # Matching is exact only within a propensity bin, so the residual shrinks as the bins get finer (Cor. 1).
    residual = []
    for n_bins in (10, 50):
        neg, kept = pma_negatives(pos, propensity_bins(deg, n_bins), is_pos, np.random.default_rng(0))
        y = np.r_[np.ones(len(kept)), np.zeros(len(neg))]
        residual.append(abs(ddiceil.auroc(y, np.r_[propensity(pos[kept]), propensity(neg)]) - 0.5))
    assert residual[0] < 0.05
    assert residual[1] < residual[0]


def test_units_without_reports_are_tied():
    rng = np.random.default_rng(1)
    m = 2000
    y = (rng.random(m) < 0.4).astype(int)
    n = np.where(rng.random(m) < 0.4, 0, rng.integers(1, 50, m)).astype(float)
    logit_r = rng.normal(-3, 0.5, m)
    F = rng.binomial(n.astype(int), 1 / (1 + np.exp(-(logit_r + 0.5 * y)))).astype(float)
    for law in ("normal", "npmle"):
        lr = ddiceil.dce(y, F, n, logit_r, law=law)["lr"]
        assert np.all(lr[n == 0] == 0.0)
