"""DirectLiNGAM (Shimizu u. a. 2011) in numpy: die exogenste Größe suchen (Entropie-Näherung nach Hyvärinen/Smith 2013), deren Einfluss herausregressieren, wiederholen; dann Regression auf die Vorgänger.
Konstanten der Näherung aus dem Gedächtnis; belegt durch das Verhalten (Gauß-Eichung, Reihenfolge auf simulierten Daten). Kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K1, K2, GAMMA = 79.047, 7.4129, 0.37457
T_MIN = 3.0
GAUSS_ENTROPY = (1.0 + math.log(2.0 * math.pi)) / 2.0


def entropy(u: np.ndarray) -> float:
    """Näherung der differentiellen Entropie einer standardisierten Variable; für N(0, 1) gleich (1 + ln 2π)/2."""
    return GAUSS_ENTROPY - K1 * (np.mean(np.log(np.cosh(u))) - GAMMA) ** 2 - K2 * np.mean(u * np.exp(-u * u / 2)) ** 2


def standardize(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residuum von a nach Regression auf b."""
    return a - (np.cov(a, b)[0, 1] / b.var(ddof=1)) * b


def pair_difference(xi: np.ndarray, xj: np.ndarray) -> float:
    """Unterschied der gemeinsamen Entropien beider Richtungen (standardisierte Größen): H(xj) + H(r_{i|j}) − H(xi) − H(r_{j|i}); positiv spricht für xi → xj (xi ist die Ursache)."""
    ri_j, rj_i = residual(xi, xj), residual(xj, xi)
    return (entropy(xj) + entropy(standardize(ri_j))) - (entropy(xi) + entropy(standardize(rj_i)))


def search_exogenous(x: np.ndarray, remaining: list) -> int:
    scores = []
    for i in remaining:
        m = 0.0
        for j in remaining:
            if i != j:
                m += min(0.0, pair_difference(standardize(x[:, i]), standardize(x[:, j]))) ** 2
        scores.append(-m)
    return remaining[int(np.argmax(scores))]


def causal_order(x: np.ndarray) -> list:
    """Kausale Reihenfolge: jeweils die exogenste Größe der übrigen, dann deren Einfluss aus den übrigen herausregressiert."""
    work = x - x.mean(axis=0)
    remaining = list(range(x.shape[1]))
    order: list = []
    for _ in range(x.shape[1] - 1):
        m = search_exogenous(work, remaining)
        order.append(m)
        remaining.remove(m)
        for i in remaining:
            work[:, i] = residual(work[:, i], work[:, m])
    return order + remaining


@dataclass
class LingamResult:
    order: list
    edges: set
    weights: np.ndarray          # weights[i, j]: Gewicht der Kante i → j (nach der Regression auf die Vorgänger)
    tvalues: np.ndarray          # tvalues[i, j]: t-Wert dieses Gewichts (0, wo i kein Vorgänger von j ist)


def direct_lingam(x: np.ndarray, t_min: float = T_MIN) -> LingamResult:
    """Reihenfolge nach DirectLiNGAM, dann Regression jeder Größe auf ihre Vorgänger; Kanten mit |t| unter t_min werden gestrichen."""
    order = causal_order(x)
    x = x - x.mean(axis=0)
    n, d = x.shape
    edges: set = set()
    weights = np.zeros((d, d))
    tvalues = np.zeros((d, d))
    for pos in range(1, d):
        j = order[pos]
        preds = order[:pos]
        z = x[:, preds]
        beta, *_ = np.linalg.lstsq(z, x[:, j], rcond=None)
        res = x[:, j] - z @ beta
        cov = np.linalg.inv(z.T @ z) * (res @ res) / (n - z.shape[1])
        t = beta / np.sqrt(np.diag(cov))
        for k, p in enumerate(preds):
            tvalues[p, j] = t[k]
            if abs(t[k]) >= t_min:
                edges.add((p, j))
                weights[p, j] = beta[k]
    return LingamResult(order, edges, weights, tvalues)


def pair_dependence(a: np.ndarray, b: np.ndarray) -> dict:
    """Für ein Paar beide Richtungen: Abhängigkeit zwischen Regressor und Residuum, gemessen als Korrelation der Quadrate (bei richtiger Richtung null, bei nicht Gauß'schem Rauschen sonst nicht)."""
    out = {}
    for name, (u, v) in (("a → b", (a, b)), ("b → a", (b, a))):
        u0, v0 = u - u.mean(), v - v.mean()
        r = residual(v0, u0)
        out[name] = {"residual": r, "regressor": u0, "dependence": float(np.corrcoef(u0 ** 2, r ** 2)[0, 1]), "linear": float(np.corrcoef(u0, r)[0, 1])}
    out["difference"] = pair_difference(standardize(a), standardize(b))
    return out
