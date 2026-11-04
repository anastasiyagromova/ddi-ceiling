"""Deconvolution ceiling estimation (DCE), Algorithm 1 of the paper.

Each unit i (a drug pair, or a pair-event triple) has n_i co-reports, F_i of which carry the signal of a channel
(an adverse event, or the reporter's "interaction" flag), and an expected rate r_i computed from single-drug data.
The latent-excess model is

    s_i | y_i ~ G_{y_i},        F_i | s_i ~ Binomial(n_i, sigmoid(logit r_i + s_i)).

DCE fits G_0 and G_1 by marginal likelihood. The ceiling is the AUROC of the latent excess itself, which bounds
the AUROC of any statistic computed from the channel. The cross-fitted likelihood ratio is the detector (DCE-LR).
"""
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

GH_X, GH_W = np.polynomial.hermite.hermgauss(20)
GRID = np.linspace(-5, 5, 81)
BOUNDS = [(-10, 10), (-5, 3)]


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """
    Mann-Whitney AUROC with average ranks for ties.

    Parameters
    ----------
    y : np.ndarray
        0/1 labels (m,)
    score : np.ndarray
        Scores (m,); higher means more likely positive

    Returns
    -------
    float
        Area under the ROC curve
    """
    y = np.asarray(y)
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Normal latent law: s ~ N(mu, tau^2), parameterised as theta = (mu, log tau).

def normal_loglik(theta, F, n, logit_r):
    """Per-unit marginal log-likelihood under a normal latent law.

    The integral over s uses 20-node Gauss-Hermite quadrature centred at each unit's posterior mode with the
    Laplace scale, which stays accurate for units with many co-reports.
    """
    mu, log_tau = theta
    tau = np.exp(log_tau)
    s = np.full(len(F), mu, float)
    for _ in range(25):
        p = _sigmoid(logit_r + s)
        grad = F - n * p - (s - mu) / tau ** 2
        hess = -n * p * (1 - p) - 1 / tau ** 2
        s = s - np.clip(grad / hess, -2, 2)
    p = _sigmoid(logit_r + s)
    sd = 1 / np.sqrt(n * p * (1 - p) + 1 / tau ** 2)
    nodes = s[:, None] + np.sqrt(2) * sd[:, None] * GH_X[None, :]
    eta = logit_r[:, None] + nodes
    log_f = (-F[:, None] * np.logaddexp(0, -eta) - (n - F)[:, None] * np.logaddexp(0, eta)
             - 0.5 * ((nodes - mu) / tau) ** 2 - np.log(tau) - 0.5 * np.log(2 * np.pi))
    return logsumexp(np.log(GH_W)[None, :] + GH_X[None, :] ** 2 + log_f, 1) + np.log(np.sqrt(2) * sd)


def _objective(theta, F, n, logit_r, h=1e-4):
    mu, log_tau = theta
    total = lambda t: normal_loglik(t, F, n, logit_r).sum()
    g_mu = (total((mu + h, log_tau)) - total((mu - h, log_tau))) / (2 * h)
    g_tau = (total((mu, log_tau + h)) - total((mu, log_tau - h))) / (2 * h)
    return -total(theta), -np.array([g_mu, g_tau])


def fit_normal(F: np.ndarray, n: np.ndarray, logit_r: np.ndarray, start: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Fit a normal latent law by maximum marginal likelihood.

    Parameters
    ----------
    F, n, logit_r : np.ndarray
        Signal counts, co-report counts and logit of the expected rate (m,)
    start : sequence of float, optional
        Starting value (mu, log tau); by default the best of four starts is kept

    Returns
    -------
    np.ndarray
        Estimate of (mu, log tau)
    """
    f = lambda th: _objective(th, F, n, logit_r)
    if start is not None:
        return minimize(f, np.array(start, float), jac=True, method="L-BFGS-B", bounds=BOUNDS).x
    expected = n / (1 + np.exp(-logit_r))
    ok = expected >= 1
    m = float(np.median(np.log((F + .5) / (expected + .5))[ok])) if ok.any() else 0.0
    fits = [minimize(f, np.array(t, float), jac=True, method="L-BFGS-B", bounds=BOUNDS)
            for t in ((0.0, 0.0), (m, 0.0), (m, -1.0), (m, 1.0))]
    return min(fits, key=lambda r: r.fun).x


def normal_ceiling(theta0: np.ndarray, theta1: np.ndarray) -> float:
    """
    Ceiling for normal latent laws: the AUROC of the latent excess itself.

    Parameters
    ----------
    theta0, theta1 : np.ndarray
        (mu, log tau) of the laws of negatives and positives

    Returns
    -------
    float
        Phi((mu1 - mu0) / sqrt(tau0^2 + tau1^2))
    """
    return float(norm.cdf((theta1[0] - theta0[0]) / np.sqrt(np.exp(2 * theta0[1]) + np.exp(2 * theta1[1]))))


# Nonparametric latent law: Kiefer-Wolfowitz NPMLE on a fixed grid.

def grid_loglik(F, n, logit_r, grid=GRID):
    """Binomial log-likelihood of each unit at each grid value of s (up to a constant per unit)."""
    eta = logit_r[:, None] + grid[None, :]
    return -F[:, None] * np.logaddexp(0, -eta) - (n - F)[:, None] * np.logaddexp(0, eta)


def fit_npmle(LL: np.ndarray, iters: int = 500, w0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit a nonparametric latent law (Kiefer-Wolfowitz NPMLE) on a fixed grid by EM.

    The problem is convex, so any start reaches the optimum.

    Parameters
    ----------
    LL : np.ndarray
        Log-likelihood of each unit at each grid value (m, n_grid), from grid_loglik
    iters : int
        Number of EM iterations
    w0 : np.ndarray, optional
        Starting weights (n_grid,)

    Returns
    -------
    np.ndarray
        Grid weights summing to one (n_grid,)
    """
    L = np.exp(LL - LL.max(1, keepdims=True))
    w = np.full(LL.shape[1], 1 / LL.shape[1]) if w0 is None else w0.copy()
    for _ in range(iters):
        p = L * w[None, :]
        p /= p.sum(1, keepdims=True)
        w = p.mean(0)
    return w


def _mixture_loglik(LL, w):
    top = LL.max(1)
    return np.log(np.exp(LL - top[:, None]) @ w + 1e-300) + top


def npmle_ceiling(w0: np.ndarray, w1: np.ndarray) -> float:
    """
    Ceiling for two discrete latent laws on the same grid (ties count one half).

    Parameters
    ----------
    w0, w1 : np.ndarray
        Grid weights of the laws of negatives and positives (n_grid,)

    Returns
    -------
    float
        AUROC of the latent excess
    """
    below = np.concatenate([[0], np.cumsum(w0)[:-1]])
    return float(np.sum(w1 * (below + 0.5 * w0)))


def dce(y: np.ndarray, F: np.ndarray, n: np.ndarray, logit_r: np.ndarray, law: str = "normal", seed: int = 0) -> Dict:
    """
    Deconvolution ceiling estimation (Algorithm 1) for one report channel.

    Parameters
    ----------
    y : np.ndarray
        0/1 labels from a curated answer key (m,)
    F : np.ndarray
        Number of co-reports that carry the signal (m,)
    n : np.ndarray
        Number of co-reports (m,)
    logit_r : np.ndarray
        Logit of the rate expected from the two drugs' single-drug data (m,)
    law : str
        "normal" or "npmle"
    seed : int
        Seed of the two-fold cross-fitting split

    Returns
    -------
    dict
        ceiling : AUROC of the latent excess, the best AUROC of any statistic of the channel
        auroc_data : AUROC of the cross-fitted likelihood ratio (DCE-LR) on these data
        lr : DCE-LR scores (m,); exactly 0 for units with n = 0
        law0, law1 : fitted laws of negatives and positives
    """
    y = np.asarray(y)
    F, n, logit_r = (np.asarray(v, float) for v in (F, n, logit_r))
    fold = np.random.default_rng(seed).integers(0, 2, len(y))
    lr = np.zeros(len(y))
    if law == "normal":
        t0 = fit_normal(F[y == 0], n[y == 0], logit_r[y == 0])
        t1 = fit_normal(F[y == 1], n[y == 1], logit_r[y == 1])
        for k in (0, 1):
            tr, te = fold != k, fold == k
            u0 = fit_normal(F[tr & (y == 0)], n[tr & (y == 0)], logit_r[tr & (y == 0)], t0)
            u1 = fit_normal(F[tr & (y == 1)], n[tr & (y == 1)], logit_r[tr & (y == 1)], t1)
            lr[te] = normal_loglik(u1, F[te], n[te], logit_r[te]) - normal_loglik(u0, F[te], n[te], logit_r[te])
        lr[n == 0] = 0.0   # no reports, no evidence: tie these units exactly instead of by rounding noise
        return dict(ceiling=normal_ceiling(t0, t1), auroc_data=auroc(y, lr), lr=lr, law0=t0, law1=t1)
    if law == "npmle":
        LL = grid_loglik(F, n, logit_r)
        w0, w1 = fit_npmle(LL[y == 0]), fit_npmle(LL[y == 1])
        for k in (0, 1):
            tr, te = fold != k, fold == k
            u0 = fit_npmle(LL[tr & (y == 0)], 200, w0)
            u1 = fit_npmle(LL[tr & (y == 1)], 200, w1)
            lr[te] = _mixture_loglik(LL[te], u1) - _mixture_loglik(LL[te], u0)
        lr[n == 0] = 0.0
        return dict(ceiling=npmle_ceiling(w0, w1), auroc_data=auroc(y, lr), lr=lr, law0=w0, law1=w1)
    raise ValueError(f"unknown law {law!r}")


def forecast_auroc(y: np.ndarray, n_target: np.ndarray, logit_r: np.ndarray, theta0: np.ndarray, theta1: np.ndarray,
                   nsim: int = 30, rng: Optional[np.random.Generator] = None) -> float:
    """
    Forecast the AUROC of raw attribution, log((F + 1/2) / (n r + 1/2)), at other co-report counts (Prop. 2).

    The latent excess of every unit is redrawn from the fitted normal laws and F from the binomial model at the
    target exposures.

    Parameters
    ----------
    y, logit_r : np.ndarray
        Labels and logit of the expected rate (m,)
    n_target : np.ndarray
        Co-report counts to forecast at, for example 4 * n for four times the reports (m,)
    theta0, theta1 : np.ndarray
        Fitted normal laws, fit["law0"] and fit["law1"] from dce(..., law="normal")
    nsim : int
        Number of simulated data sets averaged
    rng : np.random.Generator, optional
        Random generator

    Returns
    -------
    float
        Forecast AUROC
    """
    rng = np.random.default_rng(7) if rng is None else rng
    y = np.asarray(y)
    mu = np.where(y == 1, theta1[0], theta0[0])
    tau = np.exp(np.where(y == 1, theta1[1], theta0[1]))
    r = _sigmoid(logit_r)
    out = []
    for _ in range(nsim):
        s = mu + tau * rng.standard_normal(len(y))
        F = rng.binomial(n_target.astype(np.int64), _sigmoid(logit_r + s)).astype(float)
        out.append(auroc(y, np.log((F + .5) / (n_target * r + .5))))
    return float(np.mean(out))


def stratified_auroc(y: np.ndarray, score: np.ndarray, strata: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    """
    Size-weighted mean of within-stratum AUROCs; strata with a single class are skipped.

    With strata given by the strength of a single-drug association, applied to DCE-LR scores, this is the
    shortcut-free ceiling of Algorithm 1.

    Parameters
    ----------
    y, score, strata : np.ndarray
        Labels, scores and stratum of each unit (m,)
    weight : np.ndarray, optional
        Unit weights (m,), for example from drug_bootstrap

    Returns
    -------
    float
        Stratified AUROC
    """
    y, score, strata = map(np.asarray, (y, score, strata))
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    total = wsum = 0.0
    for k in np.unique(strata):
        m = (strata == k) & (weight > 0)
        if m.sum() and y[m].min() != y[m].max():
            total += roc_auc_score(y[m], score[m], sample_weight=weight[m]) * weight[m].sum()
            wsum += weight[m].sum()
    return total / wsum if wsum else np.nan


def drug_bootstrap(a: np.ndarray, b: np.ndarray, n_drugs: int, stat: Callable[[np.ndarray], float], B: int = 2000,
                   seed: int = 0) -> List[float]:
    """
    Two-way drug-cluster bootstrap: resample drugs and weight each pair by the product of its drugs' multiplicities.

    Parameters
    ----------
    a, b : np.ndarray
        Integer indices of the two drugs of each unit (m,)
    n_drugs : int
        Number of drugs
    stat : callable
        Takes a weight vector over units (m,) and returns a number
    B : int
        Number of bootstrap resamples
    seed : int
        Random seed

    Returns
    -------
    list of float
        95% percentile interval
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        c = np.bincount(rng.integers(0, n_drugs, n_drugs), minlength=n_drugs).astype(float)
        v = stat(c[a] * c[b])
        if np.isfinite(v):
            out.append(v)
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]
