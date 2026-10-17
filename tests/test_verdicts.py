"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import lci_constants as K
import lci_evaluation as ev


def test_lingam_verdict_default_is_a_success():
    kind, text = ev.verdict_lingam(ev.analyse_lingam(ev.LingamSettings()))
    assert kind == "success" and "LiNGAM richtet 6 von 6 wahren Kanten richtig (SHD 0,0)" in text and "PC richtet 4 und lässt 2 ungerichtet (SHD 1,0)" in text and "Exzess-Kurtosis des Rauschens: 3,00" in text and "weil das Rauschen nicht Gauß'sch ist" in text


def test_lingam_verdict_gaussian_names_the_missing_order():
    kind, text = ev.verdict_lingam(ev.analyse_lingam(ev.LingamSettings(kind="Gauß")))
    assert kind == "warning" and "Exzess-Kurtosis des Rauschens: 0,00" in text and "(fast) Gauß'sch" in text and "Münzwurf" in text and "überzählig" in text


def test_lingam_verdict_for_a_strong_non_gaussian_mix_lists_the_deviation():
    for seed in range(35, 90):
        a = ev.analyse_lingam(ev.LingamSettings(t=0.7, n=300, seed=seed))
        if a.compare["LiNGAM"]["shd"] > 0:
            kind, text = ev.verdict_lingam(a)
            assert kind == "warning" and "Abweichungen:" in text and "(fast) Gauß'sch" not in text
            return
    raise AssertionError("bei t = 0,7 und n = 300 sollte gelegentlich eine Kante abweichen")


def test_pair_verdict():
    kind, text = ev.verdict_pair(ev.analyse_lingam(ev.LingamSettings()))
    assert kind == "info" and "Nachfrage → Faktor: Regressor und Residuum sind in der wahren Richtung unabhängig" in text and "umgekehrten nicht" in text
    gauss = ev.verdict_pair(ev.analyse_lingam(ev.LingamSettings(kind="Gauß")))[1]
    assert "(fast) Gauß'sch" in gauss and "nahe null" in gauss


def test_icp_verdict_exact():
    kind, text = ev.verdict_icp(ev.analyse_icp(ev.IcpSettings()))
    assert kind == "success" and "ICP findet genau die Eltern des Servicegrads: {Z, X, W}" in text and "(angenommene Mengen: " in text and "mindestens 95 % nur Eltern" in text


def test_icp_verdict_for_a_partial_result_and_the_empty_cases():
    kind, text = ev.verdict_icp(ev.analyse_icp(ev.IcpSettings(envs=K.ENV_NAMES[:4])))
    assert kind == "info" and "ICP nennt {W}, aber nicht {Z, X}" in text and "Untergrenze" in text
    kind, text = ev.verdict_icp(ev.analyse_icp(ev.IcpSettings(envs=("beobachtet",))))
    assert kind == "warning" and text.startswith("Leere Menge: Mit 1 Umgebung(en) und 1000 Beobachtungen je Umgebung") and "behauptet dann nichts" in text
    kind, text = ev.verdict_icp(ev.analyse_icp(ev.IcpSettings(envs=K.ENV_NAMES)))
    assert kind == "warning" and "der 6 Umgebungen wird der Servicegrad selbst verschoben" in text and "leere Menge" in text
    kind, text = ev.verdict_icp(ev.analyse_icp(ev.IcpSettings(envs=())))
    assert kind == "warning" and text.startswith("Keine Umgebung gewählt")


def test_icp_verdict_names_a_false_parent():
    for seed in range(35, 2000):
        b = ev.analyse_icp(ev.IcpSettings(ne=300, alpha=0.1, seed=seed))
        if not b.covered:
            kind, text = ev.verdict_icp(b)
            assert kind == "error" and "keine Eltern sind" in text and "höchstens 10 %" in text
            return
    raise AssertionError("mit α = 0,1 sollte gelegentlich ein falsches Elternteil entstehen")


def test_lingam_experiment_verdict(small_lingam):
    kind, text = ev.verdict_lingam_experiment(small_lingam)
    k = small_lingam["kinds"]
    assert kind == "info" and f"in {ev.pct(k['laplace']['LiNGAM roh']['exact'])} der Läufe den genauen DAG" in text and "(n = 500, 3 Läufe)" in text and "(1 Läufe)" in text and "Standardisiert ändert sich nichts" in text


def test_gauss_experiment_verdict(small_gauss):
    kind, text = ev.verdict_gauss_experiment(small_gauss)
    first, mid, last = small_gauss["transition"]
    assert kind == "info" and f"Exzess-Kurtosis {ev.de(first['excess'], 2)}: {ev.pct(first['correct'])} richtig gerichtet (Zufall)" in text and f"{ev.de(mid['excess'], 2)}: {ev.pct(mid['correct'])}" in text and "Mit n = 100 findet LiNGAM" in text


def test_icp_experiment_verdict(small_icp):
    kind, text = ev.verdict_icp_experiment(small_icp)
    cum = small_icp["cumulative"]
    assert kind == "info" and f"gibt ICP die leere Menge ({ev.pct(cum[0]['empty'])} der Läufe)" in text and "mit 5 Umgebungen (n_e = 500)" in text and "mit n_e = 100 in" in text and "α = 0,01 kostet Trefferquote" in text and "Gauß- und Laplace-Rauschen geben dasselbe" in text


def test_limits_verdict(small_limits):
    kind, text = ev.verdict_limits(small_limits)
    p = small_limits["prediction"]
    assert kind == "info" and "Ist die Nachfrage verborgen, gibt LiNGAM in" in text and "S → X" in text and "gibt ICP in 100 % der Läufe die leere Menge" in text and f"wahre Eltern {ev.de(p['wahre Eltern']['mean'], 2)}" in text and "enthält nur einen Teil der Eltern" in text
