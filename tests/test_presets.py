"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import lci_constants as K
import lci_evaluation as ev
import lci_presets as P


def _settings(preset):
    return ev.Settings(**preset)


def test_every_preset_has_help_and_all_keys():
    assert set(K.PRESETS) == set(K.PRESET_HELP)
    for name, p in K.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) | {"envs"}, name
        assert K.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in K.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)
        assert set(p["envs"]) <= set(K.ENV_NAMES), name


def test_first_preset_is_the_default_setting():
    p = K.PRESETS["Laplace-Rauschen (Voreinstellung)"]
    assert _settings(p) == ev.Settings() and next(iter(K.PRESETS)) == "Laplace-Rauschen (Voreinstellung)"
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_at_most_one_setting():
    assert len({tuple(sorted((k, str(v)) for k, v in p.items())) for p in K.PRESETS.values()}) == len(K.PRESETS)
    base = K.PRESETS["Laplace-Rauschen (Voreinstellung)"]
    for name, p in K.PRESETS.items():
        assert len([k for k in p if p[k] != base[k]]) <= 1, name


def test_the_presets_show_what_their_help_says():
    base_l = ev.analyse_lingam(_settings(K.PRESETS["Laplace-Rauschen (Voreinstellung)"]).lingam)
    assert base_l.edges == set(K.TRUE_EDGES) and base_l.compare["PC"]["undirected"] == 2
    uni = ev.analyse_lingam(_settings(K.PRESETS["Gleichverteiltes Rauschen"]).lingam)
    assert uni.edges == set(K.TRUE_EDGES)
    gauss = ev.analyse_lingam(_settings(K.PRESETS["Gauß-Rauschen"]).lingam)
    assert gauss.compare["LiNGAM"]["shd"] >= 3 and gauss.compare["LiNGAM"]["extra"] >= 2
    fast = ev.analyse_lingam(_settings(K.PRESETS["Fast Gauß"]).lingam)
    assert fast.excess == pytest.approx(3 * 0.25 ** 2) and fast.compare["LiNGAM"]["shd"] > 0
    few = ev.analyse_icp(_settings(K.PRESETS["Wenige Umgebungen"]).icp)
    assert few.identified == {K.W} and few.covered
    small = ev.analyse_icp(_settings(K.PRESETS["Kleine Umgebungen"]).icp)
    assert small.covered and small.identified < K.TRUE_PARENTS_OF_Y
    y = ev.analyse_icp(_settings(K.PRESETS["Eingriff in den Servicegrad"]).icp)
    assert y.empty and y.y_intervention and len(K.PRESETS["Eingriff in den Servicegrad"]["envs"]) == 6


def test_snap_clamps_rounds_and_maps_the_choices():
    t = P.SETTING_SPECS["t_slider"]
    assert P.snap(9.0, t) == K.T_MAX and P.snap(-1.0, t) == K.T_MIN and P.snap(0.27, t) == pytest.approx(0.25) and P.snap(0.28, t) == pytest.approx(0.3)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == K.N_MIN and P.snap(10 ** 6, n) == K.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    k = P.SETTING_SPECS["kind_radio"]
    assert P.snap(0, k) == "Laplace" and P.snap(2, k) == "Gauß" and P.snap(9, k) == "Gauß" and P.snap(-3, k) == "Laplace" and P.url_value("kind_radio", "gleichverteilt") == 1 and P.url_value("t_slider", 0.5) == 0.5
    a = P.SETTING_SPECS["alpha_slider"]
    assert P.snap(0, a) == 0.01 and P.snap(1, a) == 0.05 and P.snap(7, a) == 0.1


def test_environment_mask_round_trip():
    assert P.envs_to_mask(K.DEFAULT_ENVS) == 31 and P.envs_to_mask(K.ENV_NAMES) == 63 and P.envs_to_mask(()) == 0 and P.envs_to_mask(("Wetter verschoben",)) == 8
    assert P.mask_to_envs(31) == K.DEFAULT_ENVS and P.mask_to_envs(63) == K.ENV_NAMES and P.mask_to_envs(0) == () and P.mask_to_envs(-5) == () and P.mask_to_envs(10 ** 6) == K.ENV_NAMES
    for mask in range(64):
        assert P.envs_to_mask(P.mask_to_envs(mask)) == mask
    assert set(P.ENV_KEYS) == set(K.ENV_NAMES) and len(set(P.ENV_KEYS.values())) == 6


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (K.N_MIN, K.N_MAX) and P.bounds("seed_input") == (0, K.SEED_MAX) and P.bounds("t_slider") == (K.T_MIN, K.T_MAX) and P.bounds("alpha_slider") == (0, 2)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"k", "t", "n", "ne", "a", "seed"} and P.ENV_PARAM == "e" and P.ENV_PARAM not in {s.url_param for s in P.SETTING_SPECS.values()}
    assert K.DEFAULT_N in range(K.N_MIN, K.N_MAX + 1, K.N_STEP) and K.DEFAULT_NE in range(K.NE_MIN, K.NE_MAX + 1, K.NE_STEP) and K.DEFAULT_ALPHA in K.ALPHAS and K.DEFAULT_KIND == K.KINDS[0]


def test_experiment_constants_are_consistent():
    assert K.EXP1_REPS >= 30 and K.EXP1_NOTEARS_REPS >= 10 and K.EXP2_TS == tuple(sorted(K.EXP2_TS)) and K.EXP2_TS[0] == 0.0 and K.EXP2_TS[-1] == 1.0 and 0.5 in K.EXP2_TS and K.EXP2_NS == tuple(sorted(K.EXP2_NS))
    assert K.EXP3_NES == tuple(sorted(K.EXP3_NES)) and K.EXP3_NE in K.EXP3_NES and 0.01 in K.EXP3_ALPHAS and 0.05 in K.EXP3_ALPHAS and K.EXP4_TRAIN_ENVS == 4 and set(K.EXP4_TEST_ENV) <= {"s_shift", "w_shift", "z_scale", "do_x", "y_shift"}
