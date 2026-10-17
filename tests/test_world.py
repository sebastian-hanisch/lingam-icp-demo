"""Welt, Rauschfamilien und Umgebungen: Varianz 1, Exzess-Kurtosis 3 t² und −1,2 t², Strukturgleichungen von Hand, Eingriffe wirken auf die richtigen Größen."""

import numpy as np
import pytest

import lci_constants as K
import lci_world as W
from kausal_generator import SplitMix64


def _excess(v):
    v = v - v.mean()
    return float((v ** 4).mean() / v.var() ** 2 - 3)


def test_noise_has_unit_variance_and_the_stated_kurtosis():
    for kind, base in (("gauss", 0.0), ("laplace", 3.0), ("uniform", -1.2)):
        for t in (0.5, 1.0):
            v = W.noise(kind, SplitMix64(2), 400000, t)
            assert v.mean() == pytest.approx(0.0, abs=0.01) and v.var() == pytest.approx(1.0, abs=0.02)
            expected = 0.0 if kind == "gauss" else base * t * t
            assert _excess(v) == pytest.approx(expected, abs=0.08 if kind == "laplace" else 0.05)
            assert W.excess_kurtosis(kind, t) == pytest.approx(expected)


def test_noise_is_deterministic_and_rejects_unknown_kinds():
    assert np.array_equal(W.noise("laplace", SplitMix64(3), 50), W.noise("laplace", SplitMix64(3), 50)) and not np.array_equal(W.noise("laplace", SplitMix64(3), 50), W.noise("laplace", SplitMix64(4), 50))
    assert np.array_equal(W.noise("laplace", SplitMix64(3), 50, 1.0), W.noise("laplace", SplitMix64(3), 50))
    with pytest.raises(ValueError):
        W.noise("cauchy", SplitMix64(1), 10)


def test_uniform_noise_is_bounded_and_laplace_is_not():
    u = W.noise("uniform", SplitMix64(5), 100000)
    assert abs(u).max() <= 3 ** 0.5 + 1e-12 and abs(W.noise("laplace", SplitMix64(5), 100000)).max() > 3 ** 0.5


def test_structural_equations_by_hand():
    """Mit Gauß-Rauschen und großer Stichprobe: Varianzen und Kovarianzen der Welt (Einheitsrauschen)."""
    x = W.simulate(300000, SplitMix64(1), "gauss")
    cov = np.cov(x, rowvar=False)
    assert np.diag(cov) == pytest.approx([1.0, 1.64, 2.0496, 3.6329, 1.0, 0.64 * 3.6329 + 1.0], abs=0.05)
    assert cov[K.S, K.Z] == pytest.approx(0.8, abs=0.02) and cov[K.Z, K.X] == pytest.approx(1.312, abs=0.03) and cov[K.W, K.X] == pytest.approx(0.0, abs=0.02) and cov[K.Y, K.C_] == pytest.approx(0.8 * 3.6329, abs=0.05)


def test_environments_act_on_the_right_columns():
    base = W.simulate(100000, SplitMix64(7), "gauss")
    shifted_s = W.simulate(100000, SplitMix64(7), "gauss", env=K.ENV_SPECS["Saison verschoben"])
    assert shifted_s[:, K.S].mean() - base[:, K.S].mean() == pytest.approx(2.0, abs=1e-9) and shifted_s[:, K.Z].mean() == pytest.approx(0.8 * 2.0, abs=0.03) and shifted_s[:, K.W].mean() == pytest.approx(0.0, abs=0.02)
    shifted_w = W.simulate(100000, SplitMix64(7), "gauss", env=K.ENV_SPECS["Wetter verschoben"])
    assert shifted_w[:, K.W].mean() == pytest.approx(2.0, abs=0.02) and shifted_w[:, K.Y].mean() == pytest.approx(0.6 * 2.0, abs=0.03) and shifted_w[:, K.S].mean() == pytest.approx(0.0, abs=0.02)
    do_x = W.simulate(100000, SplitMix64(7), "gauss", env=K.ENV_SPECS["Faktor zugeteilt"])
    assert do_x[:, K.X].var() == pytest.approx(4.0, abs=0.1) and np.corrcoef(do_x[:, K.X], do_x[:, K.Z])[0, 1] == pytest.approx(0.0, abs=0.02) and do_x[:, K.Z].var() == pytest.approx(base[:, K.Z].var(), abs=0.05)
    streu = W.simulate(100000, SplitMix64(7), "gauss", env=K.ENV_SPECS["Nachfrage streut"])
    assert streu[:, K.Z].var() == pytest.approx(0.64 + 4.0, abs=0.08)
    y_env = W.simulate(100000, SplitMix64(7), "gauss", env=K.ENV_SPECS["Eingriff in den Servicegrad"])
    assert y_env[:, K.Y].mean() == pytest.approx(2.0, abs=0.03) and y_env[:, K.Z].mean() == pytest.approx(0.0, abs=0.02) and y_env[:, K.C_].mean() == pytest.approx(0.8 * 2.0, abs=0.03)


def test_make_environments_labels_and_streams():
    data, env = W.make_environments(K.ENV_NAMES[:3], 40, 100)
    assert data.shape == (120, 6) and list(np.unique(env)) == [0, 1, 2] and (env == 1).sum() == 40
    again, _ = W.make_environments(K.ENV_NAMES[:3], 40, 100)
    assert np.array_equal(data, again)
    custom, labels = W.make_environments([{}, {"s_shift": 1.0}], 10, 5)
    assert custom.shape == (20, 6) and set(labels) == {0, 1}
    assert not np.array_equal(data[:40], data[40:80])                                                      # eigene Ströme je Umgebung


def test_constants_are_consistent():
    assert set(K.COEF) == set(K.TRUE_EDGES) and K.TRUE_PARENTS_OF_Y == {p for p, c in K.TRUE_EDGES if c == K.Y} and K.TARGET == K.Y and set(K.ENV_SPECS) == set(K.ENV_NAMES) == set(K.ENV_HELP)
    assert K.DEFAULT_ENVS == K.ENV_NAMES[:5] and "Eingriff in den Servicegrad" not in K.DEFAULT_ENVS and len(K.NAMES) == len(K.LONG) == 6 and set(K.KIND_KEYS) == set(K.KINDS)
