"""Auswertung: Kantenurteile, die zwei Analysen, die vier Experimente in kleinem Maßstab (Struktur und Identitäten)."""

import numpy as np
import pytest

import lci_constants as K
import lci_evaluation as ev


def test_number_formats():
    assert ev.de(1.5) == "1,50" and ev.de(-0.25, 3) == "−0,250" and ev.signed(0.5) == "+0,50" and ev.signed(-0.5) == "−0,50" and ev.pct(0.123, 1) == "12,3 %" and ev._n(10000) == "10 000"
    assert ev.names_of({K.Z, K.X}) == "{Z, X}" and ev.names_of(set()) == "{}" and ev.names_of([K.W, K.S]) == "{S, W}"


def test_settings_split_into_the_two_analyses():
    s = ev.Settings(kind="Gauß", t=0.5, n=500, ne=300, alpha=0.01, envs=K.ENV_NAMES[:3], seed=7)
    assert s.lingam == ev.LingamSettings("Gauß", 0.5, 500, 7) and s.icp == ev.IcpSettings("Gauß", 0.5, 300, 0.01, K.ENV_NAMES[:3], 7)
    assert s.lingam.key == "gauss" and ev.IcpSettings(kind="gleichverteilt").key == "uniform" and ev.LingamSettings().key == "laplace" and ev.Settings().lingam == ev.LingamSettings() and ev.Settings().icp == ev.IcpSettings()


def test_edge_status_covers_every_kind():
    truth = {(0, 1), (1, 2), (2, 3)}
    got = {(a, b): st for a, b, st in ev.edge_status({(0, 1), (2, 1), (0, 3)}, truth)}
    assert got[(0, 1)] == "richtig" and got[(2, 1)] == "umgekehrt" and got[(0, 3)] == "überzählig" and got[(2, 3)] == "fehlt" and (1, 2) not in got
    assert all(st == "richtig" for _, _, st in ev.edge_status(truth, truth))


def test_lingam_analysis_of_the_default_setting():
    a = ev.analyse_lingam(ev.LingamSettings())
    assert a.edges == set(K.TRUE_EDGES) and all(st == "richtig" for _, _, st in a.status) and a.excess == 3.0 and sorted(a.order) == list(range(6))
    assert a.compare["LiNGAM"]["shd"] == 0 and a.compare["PC"]["undirected"] == 2 and a.compare["PC"]["correct"] == 4 and a.compare["NOTEARS roh"]["shd"] == 0 and a.compare["NOTEARS standardisiert"]["correct"] <= 3
    assert set(a.compare) == set(ev.METHODS) and a.pair["a → b"]["dependence"] == pytest.approx(0.0, abs=0.08) and a.pair["b → a"]["dependence"] > 0.15 and a.weights[K.S, K.Z] == pytest.approx(0.8, abs=0.1)


def test_lingam_analysis_of_gaussian_noise_is_wrong_and_scale_free():
    a = ev.analyse_lingam(ev.LingamSettings(kind="Gauß"))
    assert a.excess == 0.0 and a.compare["LiNGAM"]["shd"] >= 3 and a.compare["LiNGAM"]["extra"] >= 2 and abs(a.pair["a → b"]["dependence"]) < 0.15 and abs(a.pair["b → a"]["dependence"]) < 0.15
    mix = ev.analyse_lingam(ev.LingamSettings(t=0.25))
    assert mix.excess == pytest.approx(0.1875) and mix.compare["LiNGAM"]["shd"] > 0


def test_icp_analysis_of_the_default_setting():
    b = ev.analyse_icp(ev.IcpSettings())
    assert b.identified == K.TRUE_PARENTS_OF_Y and b.exact and b.covered and not b.empty and not b.y_intervention and len(b.table) == 32 and b.table[0][1] >= b.table[1][1]
    assert sum(1 for _, _, acc in b.table if acc) == len(b.result.accepted) and set(b.share) == set(b.result.others) and all(b.share[j] == 1.0 for j in K.TRUE_PARENTS_OF_Y) and b.share[K.S] <= 1.0
    assert ev.analyse_icp(ev.IcpSettings()).identified == b.identified


def test_icp_analysis_limits():
    assert ev.analyse_icp(ev.IcpSettings(envs=("beobachtet",))).empty and ev.analyse_icp(ev.IcpSettings(envs=())).empty and ev.verdict_icp(ev.analyse_icp(ev.IcpSettings(envs=())))[1].startswith("Keine Umgebung gewählt")
    y = ev.analyse_icp(ev.IcpSettings(envs=K.ENV_NAMES))
    assert y.y_intervention and y.empty and y.covered and not y.exact and y.result.accepted == []
    few = ev.analyse_icp(ev.IcpSettings(envs=K.ENV_NAMES[:4]))
    assert few.identified == {K.W} and few.covered and not few.exact
    small = ev.analyse_icp(ev.IcpSettings(ne=300))
    assert small.covered and small.identified <= K.TRUE_PARENTS_OF_Y and small.identified != K.TRUE_PARENTS_OF_Y


def test_lingam_experiment_structure(small_lingam):
    r = small_lingam
    assert r["reps"] == 3 and r["notears_reps"] == 1 and r["n"] == 500 and set(r["kinds"]) == {"laplace", "uniform", "gauss"} and set(r["notears"]) == {"NOTEARS roh", "NOTEARS standardisiert"}
    for kind, ms in r["kinds"].items():
        assert set(ms) == {"LiNGAM roh", "LiNGAM standardisiert", "PC"}
        for m, x in ms.items():
            assert x["correct"] + x["reversed"] + x["undirected"] <= 1.0 + 1e-12 and 0 <= x["exact"] <= 1 and x["shd"] >= 0 and x["extra"] >= 0 and x["missing"] >= 0
        assert ms["LiNGAM roh"] == ms["LiNGAM standardisiert"] or kind == "gauss" or ms["LiNGAM roh"]["shd"] == pytest.approx(ms["LiNGAM standardisiert"]["shd"], abs=1.5)
    assert r["kinds"]["gauss"]["LiNGAM roh"]["exact"] <= 1 / 3 + 1e-9 and r["kinds"]["laplace"]["LiNGAM roh"]["exact"] >= 0.66 and r["kinds"]["laplace"]["PC"]["exact"] == 0.0 and r["kinds"]["laplace"]["PC"]["undirected"] > 0.2


def test_gauss_experiment_structure(small_gauss):
    r = small_gauss
    assert r["reps"] == 3 and [x["t"] for x in r["transition"]] == [0.0, 0.5, 1.0] and [x["excess"] for x in r["transition"]] == pytest.approx([0.0, 0.75, 3.0])
    assert r["transition"][0]["exact"] == 0.0 and r["transition"][-1]["correct"] > r["transition"][0]["correct"]
    assert [(x["kind"], x["n"]) for x in r["sample"]] == [("laplace", 100), ("laplace", 300), ("uniform", 100), ("uniform", 300)] and all(0 <= x["exact"] <= 1 for x in r["sample"])


def test_icp_experiment_structure_and_guarantee(small_icp):
    r = small_icp
    assert r["reps"] == 3 and r["n_e"] == 500 and [x["envs"] for x in r["cumulative"]] == [1, 2, 3, 4, 5] and [x["ne"] for x in r["sample"]] == [100, 300] and [x["alpha"] for x in r["alpha"]] == [0.01, 0.05] and [x["kind"] for x in r["kinds"]] == ["gauss", "laplace"]
    assert r["cumulative"][0]["empty"] == 1.0 and r["cumulative"][1]["empty"] == 1.0 and r["cumulative"][0]["exact"] == 0.0
    for x in r["cumulative"] + r["sample"] + r["alpha"] + r["kinds"]:
        assert 0 <= x["coverage"] <= 1 and 0 <= x["exact"] <= x["coverage"] + 1e-12 and x["false"] == pytest.approx(1 - x["coverage"]) and x["size"] <= 3 + 1e-12


def test_limits_experiment_structure(small_limits):
    r = small_limits
    assert r["reps"] == 3 and r["hidden"]["share"] >= 0.66 and r["hidden"]["main"][0].startswith("S → X") and r["y_intervention"]["empty"] == 1.0 and 0 <= r["accepted_with_c"] <= 1 and r["accepted_count"] >= 1
    p = r["prediction"]
    assert set(p) == {"alle Größen", "ohne Beschwerden", "wahre Eltern", "ICP-Schnittmenge", "nur Mittel"}
    assert p["wahre Eltern"]["mean"] == pytest.approx(1.0, abs=0.1) and p["ohne Beschwerden"]["mean"] == pytest.approx(1.0, abs=0.1) and p["alle Größen"]["mean"] > p["wahre Eltern"]["mean"] + 0.4 and p["nur Mittel"]["mean"] > 2.5
    assert p["ICP-Schnittmenge"]["mean"] > p["alle Größen"]["mean"]
    assert np.isfinite([v["sd"] for v in p.values()]).all()
