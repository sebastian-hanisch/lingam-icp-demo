"""Invariante Kausalvorhersage (Peters/Bühlmann/Meinshausen 2016) in numpy: für jede Menge S der übrigen Größen die gepoolte Regression von Y auf X_S; die Residuen müssen in allen Umgebungen gleich verteilt sein
(Mittelwert: einfaktorielle Varianzanalyse, Streuung: Levene-/Brown-Forsythe-Test, Bonferroni über beide); die Schnittmenge aller angenommenen Mengen ist die Aussage. F-Tests über die unvollständige Betafunktion. Kein Streamlit."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

TINY = 1e-300
BETA_STEPS = 300


def betacf(a: float, b: float, x: float) -> float:
    """Kettenbruch der unvollständigen Betafunktion (Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > TINY else TINY)
    h = d
    for m in range(1, BETA_STEPS):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > TINY else TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > TINY else TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > TINY else TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > TINY else TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularisierte unvollständige Betafunktion I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln_front) * betacf(a, b, x) / a
    return 1.0 - math.exp(ln_front) * betacf(b, a, 1.0 - x) / b


def f_sf(f: float, d1: float, d2: float) -> float:
    """Überlebensfunktion P(F > f) der F-Verteilung mit d1 und d2 Freiheitsgraden."""
    if f <= 0.0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def anova_p(values: np.ndarray, groups: np.ndarray) -> float:
    """p-Wert der einfaktoriellen Varianzanalyse (gleiche Gruppenmittel); eine Gruppe oder keine Streuung innerhalb: 1."""
    labels = np.unique(groups)
    k, n = len(labels), len(values)
    if k < 2:
        return 1.0
    grand = values.mean()
    ss_b = sum(((values[groups == g].mean() - grand) ** 2) * (groups == g).sum() for g in labels)
    ss_w = sum(((values[groups == g] - values[groups == g].mean()) ** 2).sum() for g in labels)
    if ss_w <= 0.0:
        return 1.0
    return f_sf((ss_b / (k - 1)) / (ss_w / (n - k)), k - 1, n - k)


def invariance_p(y: np.ndarray, xs: np.ndarray, env: np.ndarray) -> float:
    """p-Wert für „Y gegeben X_S ist in allen Umgebungen gleich“: gepoolte Regression (mit Achsenabschnitt), Residuen auf gleichen Mittelwert (ANOVA) und gleiche Streuung (Levene), Bonferroni über beide."""
    z = np.column_stack([np.ones(len(y))] + ([xs] if xs.size else []))
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    res = y - z @ beta
    p_mean = anova_p(res, env)
    dev = np.empty_like(res)
    for g in np.unique(env):
        dev[env == g] = np.abs(res[env == g] - np.median(res[env == g]))
    p_var = anova_p(dev, env)
    return min(1.0, 2.0 * min(p_mean, p_var))


@dataclass
class IcpResult:
    target: int
    others: list
    intersection: set              # identifizierte Eltern (Schnittmenge aller angenommenen Mengen)
    accepted: list                 # angenommene Mengen (Tupel von Spalten)
    p_values: dict                 # Menge (Tupel) → p-Wert
    alpha: float


def icp(data: np.ndarray, env: np.ndarray, target: int, alpha: float = 0.05) -> IcpResult:
    """ICP für eine Zielgröße: Schnittmenge aller Mengen S ⊆ übrige Größen mit p-Wert über α. Ist keine Menge angenommen, ist die Schnittmenge leer."""
    others = [j for j in range(data.shape[1]) if j != target]
    p_values: dict = {}
    accepted = []
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            xs = data[:, list(subset)] if subset else np.empty((len(data), 0))
            p = invariance_p(data[:, target], xs, env)
            p_values[subset] = p
            if p > alpha:
                accepted.append(subset)
    inter: set = set(accepted[0]) if accepted else set()
    for s in accepted[1:]:
        inter &= set(s)
    return IcpResult(target, others, inter, accepted, p_values, alpha)
