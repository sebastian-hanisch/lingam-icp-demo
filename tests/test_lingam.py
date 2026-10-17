"""DirectLiNGAM von Grund auf: Entropie-Näherung von Hand, Reihenfolge und Kanten auf simulierten Daten, Skalenfreiheit, Gauß-Rauschen, Paar."""

import math

import numpy as np
import pytest

import lci_constants as K
import lci_lingam as G
import lci_world as W
import nts_baselines as B
from kausal_generator import SplitMix64


def test_entropy_is_calibrated_at_the_gaussian():
    assert G.GAUSS_ENTROPY == pytest.approx(1.4189, abs=5e-5) and G.GAUSS_ENTROPY == pytest.approx((1 + math.log(2 * math.pi)) / 2)
    assert G.entropy(SplitMix64(1).normal(400000)) == pytest.approx(G.GAUSS_ENTROPY, abs=2e-3)


def test_entropy_is_lower_for_non_gaussian_variables():
    lap = G.standardize(W.noise("laplace", SplitMix64(2), 100000))
    uni = G.standardize(W.noise("uniform", SplitMix64(3), 100000))
    gau = G.standardize(SplitMix64(4).normal(100000))
    assert G.entropy(lap) < G.entropy(gau) - 0.02 and G.entropy(uni) < G.entropy(gau) - 0.02
    # wahre Entropien bei Varianz 1: Gauß 1,4189 > Laplace 1,3466 > gleichverteilt 1,2425; die Näherung bleibt unterhalb von Gauß
    assert 1 + math.log(math.sqrt(2)) == pytest.approx(1.3466, abs=5e-5) and math.log(2 * math.sqrt(3)) == pytest.approx(1.2425, abs=5e-5)


def test_standardize_and_residual():
    v = SplitMix64(5).normal(1000) * 3 + 7
    z = G.standardize(v)
    assert z.mean() == pytest.approx(0.0, abs=1e-12) and z.std() == pytest.approx(1.0)
    a = SplitMix64(6).normal(5000)
    b = 0.7 * a + SplitMix64(7).normal(5000)
    r = G.residual(b - b.mean(), a - a.mean())
    assert np.cov(r, a)[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_pair_difference_prefers_the_true_direction_for_non_gaussian_noise():
    for kind in ("laplace", "uniform"):
        a = W.noise(kind, SplitMix64(8), 20000)
        b = 0.8 * a + W.noise(kind, SplitMix64(9), 20000)
        d = G.pair_difference(G.standardize(a), G.standardize(b))
        assert d > 0.01 and G.pair_difference(G.standardize(b), G.standardize(a)) == pytest.approx(-d, abs=1e-12)
    a, b = SplitMix64(10).normal(20000), None
    b = 0.8 * a + SplitMix64(11).normal(20000)
    assert abs(G.pair_difference(G.standardize(a), G.standardize(b))) < 0.01


def test_pair_dependence_is_zero_in_the_true_direction_only():
    a = W.noise("laplace", SplitMix64(12), 50000)
    b = 0.8 * a + W.noise("laplace", SplitMix64(13), 50000)
    p = G.pair_dependence(a, b)
    assert abs(p["a → b"]["dependence"]) < 0.03 and p["b → a"]["dependence"] > 0.15 and abs(p["a → b"]["linear"]) < 1e-9 and abs(p["b → a"]["linear"]) < 1e-9 and p["difference"] > 0
    g = G.pair_dependence(SplitMix64(14).normal(50000), SplitMix64(15).normal(50000))
    assert abs(g["a → b"]["dependence"]) < 0.03 and abs(g["b → a"]["dependence"]) < 0.03


def test_the_causal_order_of_the_world_with_non_gaussian_noise():
    for kind in ("laplace", "uniform"):
        x = W.simulate(1000, SplitMix64(35), kind)
        order = G.causal_order(x)
        pos = {v: i for i, v in enumerate(order)}
        assert all(pos[a] < pos[b] for a, b in K.TRUE_EDGES), (kind, order)


def test_lingam_finds_the_dag_for_non_gaussian_noise_and_ignores_the_scale():
    for kind in ("laplace", "uniform"):
        x = W.simulate(1000, SplitMix64(35), kind)
        res = G.direct_lingam(x)
        assert res.edges == set(K.TRUE_EDGES), kind
        assert G.direct_lingam(B.standardize(x)).edges == res.edges and G.direct_lingam(x * np.array([1, 5, 0.2, 3, 9, 0.5])).edges == res.edges
        for a, b in K.TRUE_EDGES:
            assert res.weights[a, b] == pytest.approx(K.COEF[(a, b)], abs=0.12), (a, b)


def test_lingam_cannot_order_gaussian_data():
    shd = []
    for r in range(8):
        edges = G.direct_lingam(W.simulate(1000, SplitMix64(100 + r), "gauss")).edges
        shd.append(len(edges ^ set(K.TRUE_EDGES)))
    assert min(shd) > 0 and np.mean(shd) > 3                                                  # nie genau, im Mittel mehrere falsche Kanten


def test_lingam_result_structure_and_pruning():
    x = W.simulate(1000, SplitMix64(35), "laplace")
    res = G.direct_lingam(x)
    assert sorted(res.order) == list(range(6)) and res.weights.shape == (6, 6) and all(res.weights[a, b] != 0 for a, b in res.edges) and np.count_nonzero(res.weights) == len(res.edges)
    assert len(G.direct_lingam(x, t_min=0.0).edges) == 15 and G.direct_lingam(x, t_min=1e9).edges == set()


def test_independent_columns_give_no_edges():
    x = np.column_stack([W.noise("laplace", SplitMix64(20 + j), 3000) for j in range(4)])
    assert G.direct_lingam(x).edges == set()


def test_search_exogenous_picks_a_root():
    x = W.simulate(2000, SplitMix64(36), "laplace")
    x = x - x.mean(axis=0)
    assert G.search_exogenous(x, list(range(6))) in (K.S, K.W)


def test_entropy_approximation_values_pin_the_constants():
    """Die Näherung ist grob (gleichverteilt 1,36 statt wahr 1,24), ordnet aber Gauß über beide; die Werte legen die Konstanten und Exponenten fest."""
    expected = {("laplace", 1.0): 1.3165, ("laplace", 0.5): 1.4121, ("uniform", 1.0): 1.3623, ("uniform", 0.5): 1.4165, ("gauss", 1.0): 1.4189}
    for (kind, t), value in expected.items():
        assert G.entropy(G.standardize(W.noise(kind, SplitMix64(3), 600000, t))) == pytest.approx(value, abs=0.002), (kind, t)
    assert G.entropy(np.array([0.0, 1.0, -1.0, 2.0, -2.0])) == pytest.approx(G.GAUSS_ENTROPY - G.K1 * (np.mean(np.log(np.cosh([0.0, 1.0, -1.0, 2.0, -2.0]))) - G.GAMMA) ** 2 - G.K2 * 0.0, abs=1e-12)     # ungerade Momente von symmetrischen Daten: der zweite Term ist null


def test_lingam_does_not_depend_on_the_location():
    x = W.simulate(1000, SplitMix64(35), "laplace")
    a, b = G.direct_lingam(x), G.direct_lingam(x + np.array([10.0, -5.0, 3.0, 20.0, -7.0, 1.0]))
    assert a.edges == b.edges and a.order == b.order and a.weights == pytest.approx(b.weights, abs=1e-9) and a.tvalues == pytest.approx(b.tvalues, abs=1e-6)


def test_t_values_equal_the_partial_f_statistic_of_a_refit():
    """Zweiter Weg zum t-Wert: t² = (RSS ohne die Größe − RSS voll) / (RSS voll / (n − k))."""
    x = W.simulate(1000, SplitMix64(35), "laplace")
    res = G.direct_lingam(x)
    xc = x - x.mean(axis=0)
    n = len(x)
    checked = 0
    for j in range(6):
        preds = res.order[:res.order.index(j)]
        if len(preds) < 2:
            continue
        z = xc[:, preds]
        rss_full = np.sum((xc[:, j] - z @ np.linalg.lstsq(z, xc[:, j], rcond=None)[0]) ** 2)
        for k, p in enumerate(preds):
            z0 = np.delete(z, k, axis=1)
            rss0 = np.sum((xc[:, j] - z0 @ np.linalg.lstsq(z0, xc[:, j], rcond=None)[0]) ** 2)
            f = (rss0 - rss_full) / (rss_full / (n - len(preds)))
            assert res.tvalues[p, j] ** 2 == pytest.approx(f, rel=1e-8)
            assert np.sign(res.tvalues[p, j]) == np.sign(res.weights[p, j]) or res.weights[p, j] == 0
            checked += 1
    assert checked >= 10 and res.tvalues[K.S, K.Z] > 20 and np.count_nonzero(res.tvalues) == 15


def test_pair_dependence_does_not_depend_on_the_location():
    a = W.noise("laplace", SplitMix64(12), 20000)
    b = 0.8 * a + W.noise("laplace", SplitMix64(13), 20000)
    p, q = G.pair_dependence(a, b), G.pair_dependence(a + 7.0, b - 3.0)
    assert p["a → b"]["dependence"] == pytest.approx(q["a → b"]["dependence"], abs=1e-9) and p["b → a"]["dependence"] == pytest.approx(q["b → a"]["dependence"], abs=1e-9)


def test_entropy_of_a_skewed_variable_pins_the_second_term():
    """Bei schiefen Daten ist E[u e^(−u²/2)] nicht null: Exponentialverteilung (wahre Entropie 1,0; die Näherung gibt 1,0913), gespiegelt gleich (der Term steht quadriert)."""
    z = G.standardize(-np.log(1.0 - SplitMix64(3).uniform(600000)))
    assert G.entropy(z) == pytest.approx(1.0913, abs=0.003) and G.entropy(-z) == pytest.approx(G.entropy(z), abs=1e-12)
    assert np.mean(z * np.exp(-z * z / 2)) == pytest.approx(-0.1538, abs=0.003) and G.GAUSS_ENTROPY - G.entropy(z) > 0.3
