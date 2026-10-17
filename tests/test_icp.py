"""ICP von Grund auf: unvollständige Betafunktion und F-Test von Hand und gegen scipy, Invarianztest, Schnittmenge, Abdeckung."""

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st

import lci_constants as K
import lci_icp as I
import lci_world as W
from kausal_generator import SplitMix64


def test_betainc_by_hand_and_against_scipy():
    assert I.betainc(2.0, 3.0, 0.4) == pytest.approx(0.5248, abs=1e-12)                      # Σ_{j=2}^{4} C(4, j) 0,4^j 0,6^(4−j)
    assert 6 * 0.16 * 0.36 + 4 * 0.064 * 0.6 + 0.0256 == pytest.approx(0.5248)
    for a, b, x in ((2.0, 3.0, 0.4), (0.5, 7.5, 0.9), (30.0, 2.0, 0.95), (1.0, 1.0, 0.3), (5.0, 5.0, 0.5), (0.5, 0.5, 0.01), (200.0, 700.0, 0.2)):
        assert I.betainc(a, b, x) == pytest.approx(sp.betainc(a, b, x), abs=1e-10), (a, b, x)
    assert I.betainc(2, 3, 0.0) == 0.0 and I.betainc(2, 3, 1.0) == 1.0 and I.betainc(2, 3, -0.5) == 0.0 and I.betainc(2, 3, 1.5) == 1.0
    assert I.betainc(1.0, 1.0, 0.37) == pytest.approx(0.37) and I.betainc(3.0, 3.0, 0.5) == pytest.approx(0.5)


def test_f_survival_against_scipy():
    for f, d1, d2 in ((1.3, 4, 1500), (5.0, 2, 300), (0.4, 4, 1495), (1.5, 1, 4), (20.0, 5, 50), (0.01, 3, 100)):
        assert I.f_sf(f, d1, d2) == pytest.approx(st.f.sf(f, d1, d2), abs=1e-10), (f, d1, d2)
    assert I.f_sf(0.0, 2, 10) == 1.0 and I.f_sf(-1.0, 2, 10) == 1.0 and I.f_sf(1.5, 1, 4) == pytest.approx(0.2879, abs=5e-5)


def test_anova_p_by_hand():
    values = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    # ss_b = 3 · 0,25 · 2 = 1,5; ss_w = 2 + 2 = 4; F = (1,5 / 1) / (4 / 4) = 1,5, p = 0,2879 (t = 1,2247, vier Freiheitsgrade)
    assert I.anova_p(values, groups) == pytest.approx(0.2879, abs=5e-5)
    assert I.anova_p(values, groups) == pytest.approx(st.f_oneway(values[:3], values[3:]).pvalue, abs=1e-10)
    assert I.anova_p(values, np.zeros(6, dtype=int)) == 1.0 and I.anova_p(np.ones(6), groups) == 1.0


def test_anova_p_against_scipy_for_three_groups():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0, 1, 50), rng.normal(0.3, 1, 60), rng.normal(0, 1.2, 40)])
    g = np.repeat([0, 1, 2], [50, 60, 40])
    assert I.anova_p(v, g) == pytest.approx(st.f_oneway(v[g == 0], v[g == 1], v[g == 2]).pvalue, abs=1e-10)


def test_invariance_by_hand():
    """Y = X + ε, in der zweiten Umgebung X um 3 verschoben: ohne X nicht invariant, mit X invariant."""
    rng = SplitMix64(1)
    x = np.concatenate([rng.normal(500), rng.normal(500) + 3.0])
    y = x + 0.5 * np.concatenate([rng.normal(500), rng.normal(500)])
    env = np.repeat([0, 1], 500)
    assert I.invariance_p(y, np.empty((1000, 0)), env) < 1e-10 and I.invariance_p(y, x[:, None], env) > 0.01
    res = I.icp(np.column_stack([x, y]), env, 1, 0.05)
    assert res.intersection == {0} and res.accepted == [(0,)] and res.others == [0] and res.p_values[()] < 1e-10


def test_variance_shift_is_detected_by_the_levene_part():
    rng = SplitMix64(3)
    y = np.concatenate([rng.normal(2000), 2.0 * rng.normal(2000)])
    env = np.repeat([0, 1], 2000)
    assert I.invariance_p(y, np.empty((4000, 0)), env) < 1e-10
    same = np.concatenate([rng.normal(2000), rng.normal(2000)])
    assert I.invariance_p(same, np.empty((4000, 0)), env) > 0.01


def test_a_single_environment_accepts_everything_and_returns_the_empty_set():
    data, env = W.make_environments(["beobachtet"], 800, 5, "gauss")
    res = I.icp(data, env, K.TARGET, 0.05)
    assert len(res.accepted) == 32 and res.intersection == set() and all(p == 1.0 for p in res.p_values.values())


def test_the_world_with_all_five_environments_finds_the_parents():
    data, env = W.make_environments(K.DEFAULT_ENVS, 3000, 4000, "gauss")
    res = I.icp(data, env, K.TARGET, 0.05)
    assert res.intersection == K.TRUE_PARENTS_OF_Y and all(set(s) >= K.TRUE_PARENTS_OF_Y for s in res.accepted) and (K.Z, K.X, K.W) in [tuple(sorted(s)) for s in res.accepted]
    assert res.p_values[tuple(sorted(K.TRUE_PARENTS_OF_Y))] > 0.05 and res.p_values[(K.Z, K.X)] < 1e-3 and res.p_values[(K.X, K.W)] < 1e-3 and res.p_values[(K.Z, K.W)] < 1e-3


def test_children_are_not_invariant_predictors_on_their_own():
    data, env = W.make_environments(K.DEFAULT_ENVS, 3000, 4000, "gauss")
    res = I.icp(data, env, K.TARGET, 0.05)
    assert res.p_values[(K.C_,)] < 1e-3 and res.p_values[()] < 1e-3


def test_an_intervention_on_the_target_gives_the_empty_set():
    data, env = W.make_environments(K.ENV_NAMES, 1500, 4000, "gauss")
    res = I.icp(data, env, K.TARGET, 0.05)
    assert res.intersection == set() and res.accepted == [] and max(res.p_values.values()) < 0.05


def test_the_coverage_guarantee_holds_over_repeats():
    wrong = 0
    reps = 60
    for r in range(reps):
        data, env = W.make_environments(K.DEFAULT_ENVS, 600, 7000 + 17 * r, "gauss")
        wrong += not I.icp(data, env, K.TARGET, 0.05).intersection <= K.TRUE_PARENTS_OF_Y
    assert wrong <= 0.1 * reps                                         # Garantie: höchstens α = 0,05 (hier mit Spielraum für 60 Läufe)


def test_alpha_one_accepts_nothing_and_alpha_zero_accepts_everything():
    data, env = W.make_environments(K.DEFAULT_ENVS, 400, 11, "gauss")
    assert I.icp(data, env, K.TARGET, 0.0).intersection == set() and len(I.icp(data, env, K.TARGET, 0.0).accepted) == 32
    assert I.icp(data, env, K.TARGET, 1.0).accepted == [] and I.icp(data, env, K.TARGET, 1.0).intersection == set()


def test_a_p_value_equal_to_alpha_is_not_accepted():
    """Mit einer Umgebung ist p = 1 für jede Menge: bei α = 1 ist keine angenommen (p muss größer sein), bei α knapp darunter alle."""
    data, env = W.make_environments(["beobachtet"], 500, 3, "gauss")
    assert I.icp(data, env, K.TARGET, 1.0).accepted == [] and len(I.icp(data, env, K.TARGET, 0.999999).accepted) == 32
