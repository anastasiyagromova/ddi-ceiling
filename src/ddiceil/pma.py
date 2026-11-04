"""Propensity-matched audit (PMA), Algorithm 2 of the paper, and the negative sampler for identifiability-first training.

A benchmark that pairs each positive with random negatives can be solved by single-drug propensity alone: drugs with
many interactions are likely to appear in positives. PMA builds negatives that keep one drug of each positive and
replace the other by a drug of the same propensity, so every function of the two drugs' single-drug profiles has
AUROC 1/2 against them (Corollary 1). Whatever AUROC a model keeps against these negatives is pair-specific.
"""
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata


def propensity_bins(propensity: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Quantile bin of each drug's single-drug propensity.

    Parameters
    ----------
    propensity : np.ndarray
        One value per drug, for example its training degree (n_drugs,)
    n_bins : int
        Number of quantile bins (deciles by default)

    Returns
    -------
    np.ndarray
        Bin index 0 .. n_bins-1 of each drug (n_drugs,)
    """
    propensity = np.asarray(propensity)
    return np.floor(rankdata(propensity, method="ordinal") / (len(propensity) + 1) * n_bins).astype(int)


def pma_negatives(pos: np.ndarray, bins: np.ndarray, is_positive: Callable[[int, int], bool], rng: np.random.Generator,
                  partners: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propensity-matched negatives for a list of positive pairs (Algorithm 2).

    For a positive (a, b), one drug k is kept (chosen at random) and the other, o, is replaced by a drug q with
    bins[q] == bins[o], q != k, and (k, q) not a positive. If no q exists the roles of the two drugs are swapped;
    if there is still none the positive is dropped.

    Parameters
    ----------
    pos : np.ndarray
        Positive pairs (m, 2), integer drug indices
    bins : np.ndarray
        Propensity bin of each drug, from propensity_bins (n_drugs,)
    is_positive : callable
        is_positive(a, b) is True if (a, b) is a positive pair
    rng : np.random.Generator
        Random generator
    partners : sequence of np.ndarray, optional
        partners[k] lists the drugs co-reported with k; q is then drawn from k's partners only, so every negative
        is a real co-exposure

    Returns
    -------
    neg : np.ndarray
        Negative pairs (m_kept, 2)
    kept : np.ndarray
        Indices of the positives that were matched (m_kept,)
    """
    n_drugs = len(bins)
    neg, kept = [], []
    for idx, (a, b) in enumerate(pos):
        order = ((a, b), (b, a)) if rng.random() < 0.5 else ((b, a), (a, b))
        for keep, other in order:
            pool = np.arange(n_drugs) if partners is None else partners[keep]
            pool = pool[(bins[pool] == bins[other]) & (pool != keep)]
            pool = [q for q in pool if not is_positive(keep, q)]
            if pool:
                neg.append((keep, pool[rng.integers(len(pool))]))
                kept.append(idx)
                break
    return np.array(neg, int).reshape(-1, 2), np.array(kept, int)


def shortcut_share(auroc_random: float, auroc_pma: float) -> float:
    """
    Share of the above-chance AUROC under random negatives that is lost under PMA negatives.

    Parameters
    ----------
    auroc_random, auroc_pma : float
        AUROC of a model against the benchmark's random negatives and against PMA negatives

    Returns
    -------
    float
        (auroc_random - auroc_pma) / (auroc_random - 1/2)
    """
    return (auroc_random - auroc_pma) / (auroc_random - 0.5)


def leakage(auroc_propensity_pma: float) -> float:
    """
    Distance from chance of a propensity-only model under PMA; zero up to within-bin variation (Cor. 1).

    Parameters
    ----------
    auroc_propensity_pma : float
        AUROC of the propensity-only model against PMA negatives

    Returns
    -------
    float
        |auroc_propensity_pma - 1/2|
    """
    return abs(auroc_propensity_pma - 0.5)


class MatchedNegativeSampler:
    """Degree-matched negatives for training.

    Keeps one endpoint of each positive edge and replaces the other by a node of the same type and the same
    training-degree bin that is not linked to the kept node. Nodes that cannot be matched in `tries` draws fall
    back to random negatives.

    Parameters
    ----------
    bins : (n,) training-degree bin of each node.
    is_edge : function (u, v) -> boolean array, true where (u, v) is a training edge.
    random_negatives : function m -> (m, 2) array of random non-edges.
    types : (n,) node type for bipartite graphs (drug and target, say), or None.
    """

    def __init__(self, bins: np.ndarray, is_edge: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 random_negatives: Callable[[int], np.ndarray], rng: np.random.Generator,
                 types: Optional[np.ndarray] = None, tries: int = 10):
        self.is_edge, self.random_negatives, self.rng, self.tries = is_edge, random_negatives, rng, tries
        types = np.zeros(len(bins), int) if types is None else np.asarray(types)
        group = types * (bins.max() + 1) + bins
        self.order = np.argsort(group, kind="stable")
        self.start = np.searchsorted(group[self.order], group)
        self.size = np.bincount(group)[group]

    def __call__(self, P: np.ndarray) -> np.ndarray:
        m = len(P)
        flip = self.rng.random(m) < .5
        keep = np.where(flip, P[:, 1], P[:, 0])
        other = np.where(flip, P[:, 0], P[:, 1])
        q = np.full(m, -1)
        todo = np.arange(m)
        for _ in range(self.tries):
            o = other[todo]
            c = self.order[self.start[o] + (self.rng.random(len(todo)) * self.size[o]).astype(int)]
            ok = (c != keep[todo]) & ~self.is_edge(keep[todo], c)
            q[todo[ok]] = c[ok]
            todo = todo[~ok]
            if not len(todo):
                break
        N = np.stack([np.minimum(keep, q), np.maximum(keep, q)], 1)
        if len(todo):
            N[todo] = self.random_negatives(len(todo))
        return N


def identifiability_first_negatives(P: np.ndarray, sampler: "MatchedNegativeSampler",
                                    random_negatives: Callable[[int], np.ndarray]) -> np.ndarray:
    """
    Negatives for one training batch: half degree-matched, half random.

    Identifiability-first training uses these negatives and selects the model (early stopping) by AUROC on
    validation positives against PMA negatives.

    Parameters
    ----------
    P : np.ndarray
        Positive edges of the batch (m, 2)
    sampler : MatchedNegativeSampler
        Degree-matched sampler
    random_negatives : callable
        random_negatives(k) returns k random non-edges as a (k, 2) array

    Returns
    -------
    np.ndarray
        Negative edges (m, 2)
    """
    h = len(P) // 2
    return np.r_[sampler(P[:h]), random_negatives(len(P) - h)]
