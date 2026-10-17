"""Dauerhafte Orakel mit bekannter Wahrheit und unabhängiger zweiter Rechnung: (1) DirectLiNGAM auf Zufalls-DAGs mit nicht Gauß'schem Rauschen findet die wahre Reihenfolge und Kantenmenge;
(2) der Invarianztest von ICP stimmt mit einer Rechnung über scipy (f_oneway, levene mit Median) überein, und die Schnittmenge folgt aus den scipy-p-Werten."""

import itertools

import numpy as np
import pytest
import scipy.stats as st

import lci_constants as K
import lci_icp as I
import lci_lingam as G
import lci_world as W
import stl_world as SW
from kausal_generator import SplitMix64


def simulate_dag(b: np.ndarray, n: int, rng: SplitMix64, kind: str) -> np.ndarray:
    d = b.shape[0]
    e = np.column_stack([W.noise(kind, rng, n) for _ in range(d)])
    return e @ np.linalg.inv(np.eye(d) - b).T


def test_random_dags_with_non_gaussian_noise_are_recovered():
    rng = SplitMix64(31)
    exact = checked = 0
    for r in range(12):
        b = SW.random_dag(5, 0.5, rng)
        if np.count_nonzero(b) < 3:
            continue
        truth = {(i, j) for j in range(5) for i in range(5) if b[j, i] != 0}
        # DirectLiNGAM kennt die Spaltennummern nicht: die Spalten werden zufällig vertauscht, die Wahrheit mit
        perm = np.random.default_rng(r).permutation(5)
        x = simulate_dag(b, 2000, SplitMix64(500 + r), "laplace")[:, perm]
        inv = {int(old): new for new, old in enumerate(perm)}
        truth_p = {(inv[i], inv[j]) for i, j in truth}
        res = G.direct_lingam(x)
        pos = {v: k for k, v in enumerate(res.order)}
        assert all(pos[a] < pos[c] for a, c in truth_p), (r, res.order, truth_p)
        exact += res.edges == truth_p
        checked += 1
    assert checked >= 8 and exact >= checked - 2


def test_invariance_p_equals_the_scipy_computation():
    rng = np.random.default_rng(2)
    n = 400
    env = np.repeat([0, 1, 2], n)
    xs = rng.normal(size=(3 * n, 2))
    y = xs @ np.array([0.7, -0.4]) + rng.normal(size=3 * n) * np.where(env == 2, 1.6, 1.0) + np.where(env == 1, 0.3, 0.0)
    z = np.column_stack([np.ones(3 * n), xs])
    res = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    p_mean = st.f_oneway(*[res[env == g] for g in range(3)]).pvalue
    p_var = st.levene(*[res[env == g] for g in range(3)], center="median").pvalue
    assert I.invariance_p(y, xs, env) == pytest.approx(min(1.0, 2 * min(p_mean, p_var)), abs=1e-9)
    assert I.anova_p(np.abs(res - 0), env) == pytest.approx(st.f_oneway(*[np.abs(res)[env == g] for g in range(3)]).pvalue, abs=1e-9)


def test_levene_part_is_the_scipy_brown_forsythe_test():
    rng = np.random.default_rng(3)
    res = np.concatenate([rng.normal(size=300), 1.5 * rng.normal(size=300)])
    env = np.repeat([0, 1], 300)
    dev = np.empty_like(res)
    for g in (0, 1):
        dev[env == g] = np.abs(res[env == g] - np.median(res[env == g]))
    assert I.anova_p(dev, env) == pytest.approx(st.levene(res[env == 0], res[env == 1], center="median").pvalue, abs=1e-10)


def test_icp_intersection_equals_the_scipy_intersection():
    data, env = W.make_environments(K.DEFAULT_ENVS, 800, 6100, "gauss")
    mine = I.icp(data, env, K.TARGET, 0.05)
    others = [j for j in range(6) if j != K.TARGET]
    accepted = []
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            z = np.column_stack([np.ones(len(data))] + ([data[:, list(sub)]] if sub else []))
            res = data[:, K.TARGET] - z @ np.linalg.lstsq(z, data[:, K.TARGET], rcond=None)[0]
            groups = [res[env == g] for g in np.unique(env)]
            p = min(1.0, 2 * min(st.f_oneway(*groups).pvalue, st.levene(*groups, center="median").pvalue))
            assert mine.p_values[sub] == pytest.approx(p, abs=1e-8), sub
            if p > 0.05:
                accepted.append(sub)
    inter = set(accepted[0]) if accepted else set()
    for s in accepted[1:]:
        inter &= set(s)
    assert mine.accepted == accepted and mine.intersection == inter
