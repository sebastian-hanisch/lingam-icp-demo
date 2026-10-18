"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil); die Verfahren rechnen aber in Gleitkomma. Wo eine einzelne Kante oder ein einzelner Lauf je Plattform kippen kann (Gauß-Rauschen, in dem die
Reihenfolge eine Auswahl ist; Invarianztests nahe α; standardisiertes NOTEARS), gelten Toleranzen von zwei Läufen. Was Garantie oder Konstruktion ist, steht ausdrücklich dabei: die Abdeckung von ICP (Garantie, kein
Befund), die leere Menge bei Eingriff in die Zielgröße (Konstruktion der Methode), die Kurtosis-Formel 3 t² (Rechenidentität).
"""

import math

import numpy as np
import pytest

import lci_constants as K
import lci_evaluation as ev
import lci_icp as I
import lci_lingam as G
import lci_world as W
from kausal_generator import SplitMix64

TOL = 0.045            # zwei Läufe von 50


def test_default_sample():
    a = ev.analyse_lingam(ev.LingamSettings())
    assert a.edges == set(K.TRUE_EDGES) and a.compare["LiNGAM"]["shd"] == 0 and (a.compare["PC"]["correct"], a.compare["PC"]["undirected"], a.compare["PC"]["shd"]) == (4, 2, 1.0) and a.compare["NOTEARS roh"]["shd"] == 0
    assert a.compare["NOTEARS standardisiert"]["correct"] <= 3 and ev.verdict_lingam(a)[0] == "success" and a.excess == 3.0
    b = ev.analyse_icp(ev.IcpSettings())
    assert b.identified == K.TRUE_PARENTS_OF_Y and len(b.result.accepted) == 4 and ev.verdict_icp(b)[0] == "success"


def test_hand_values_of_the_text():
    assert G.GAUSS_ENTROPY == pytest.approx(1.4189, abs=0.00006) and 1 + math.log(math.sqrt(2)) == pytest.approx(1.3466, abs=0.00006) and math.log(2 * math.sqrt(3)) == pytest.approx(1.2425, abs=0.00006)
    assert [W.excess_kurtosis("laplace", t) for t in (0.5, 1.0)] == [0.75, 3.0] and W.excess_kurtosis("uniform", 1.0) == -1.2 and W.excess_kurtosis("gauss", 1.0) == 0.0
    assert I.f_sf(1.5, 1, 4) == pytest.approx(0.2879, abs=0.00006) and I.betainc(2.0, 3.0, 0.4) == pytest.approx(0.5248, abs=0.00006)
    assert 3 * 0.25 ** 2 == pytest.approx(0.1875) and ev.de(W.excess_kurtosis("laplace", 0.25), 2) == "0,19"


def test_the_kurtosis_formula_is_an_identity_and_the_noise_matches_it():
    """Exzess-Kurtosis 3 t² (Laplace-Mischung) folgt aus den Kumulanten: √(1 − t) Gauß + √t Laplace mit Varianz 1; die gemessene Kurtosis der Welt liegt darauf."""
    for t in (0.25, 0.5, 1.0):
        v = W.noise("laplace", SplitMix64(2), 400000, t)
        v = v - v.mean()
        assert float((v ** 4).mean() / v.var() ** 2 - 3) == pytest.approx(3 * t * t, abs=0.1)


def test_experiment_1_which_methods_leave_the_class(full_lingam):
    r = full_lingam
    k, nt = r["kinds"], r["notears"]
    assert r["reps"] == 50 and r["notears_reps"] == 12 and r["n"] == 1000 and K.EXP1_REPS == 50
    assert k["laplace"]["LiNGAM roh"]["exact"] == pytest.approx(0.94, abs=TOL) and k["laplace"]["LiNGAM roh"]["correct"] == 1.0 and k["uniform"]["LiNGAM roh"]["exact"] == pytest.approx(0.98, abs=TOL) and k["uniform"]["LiNGAM roh"]["correct"] == 1.0
    assert k["gauss"]["LiNGAM roh"]["exact"] == 0.0 and k["gauss"]["LiNGAM roh"]["correct"] == pytest.approx(0.52, abs=0.05) and k["gauss"]["LiNGAM roh"]["extra"] == pytest.approx(3.6, abs=0.4) and k["gauss"]["LiNGAM roh"]["shd"] == pytest.approx(6.48, abs=0.6)
    # skalenfrei: LiNGAM standardisiert gibt dieselben Zahlen (DirectLiNGAM standardisiert ohnehin; der Befund ist der Gegensatz zu NOTEARS)
    assert all(k[kind]["LiNGAM roh"] == k[kind]["LiNGAM standardisiert"] for kind in k)
    assert (k["laplace"]["PC"]["correct"], k["laplace"]["PC"]["undirected"], k["laplace"]["PC"]["exact"]) == pytest.approx((0.6667, 0.3233, 0.0), abs=0.0006)
    assert nt["NOTEARS roh"]["correct"] >= 0.95 and nt["NOTEARS roh"]["exact"] >= 0.9 and nt["NOTEARS standardisiert"]["correct"] == pytest.approx(0.5, abs=0.15) and nt["NOTEARS standardisiert"]["reversed"] == pytest.approx(0.28, abs=0.15) and nt["NOTEARS standardisiert"]["exact"] == 0.0
    text = ev.verdict_lingam_experiment(r)[1]
    assert "Bei Laplace-Rauschen (n = 1 000, 50 Läufe) findet LiNGAM in 94 %" in text and "bei Gauß-Rauschen in 0 % (richtig gerichtet 52 %" in text and "PC lässt 32 % der Kanten ungerichtet" in text and "(12 Läufe)" in text


def test_experiment_2_transition_and_sample(full_gauss):
    r = full_gauss
    tr = {x["t"]: x for x in r["transition"]}
    assert r["reps"] == 50 and [x["excess"] for x in r["transition"]] == pytest.approx([0.0, 0.03, 0.1875, 0.75, 1.6875, 3.0])
    # nahe Gauß sind die Richtungen Zufall (Verfahrenseigenschaft: die Reihenfolge ist nicht bestimmt), danach ein Übergang
    assert [tr[t]["correct"] for t in (0.0, 0.1, 0.25)] == pytest.approx([0.5267, 0.5467, 0.55], abs=0.05) and all(tr[t]["exact"] <= 0.05 for t in (0.0, 0.1, 0.25))
    assert tr[0.5]["correct"] == pytest.approx(0.92, abs=0.05) and tr[0.5]["exact"] == pytest.approx(0.7, abs=0.1) and tr[0.75]["correct"] == 1.0 and tr[0.75]["exact"] >= 0.95 and tr[1.0]["correct"] == 1.0 and tr[1.0]["exact"] == pytest.approx(0.9, abs=0.08)
    s = {(x["kind"], x["n"]): x for x in r["sample"]}
    for key, exact in {("laplace", 100): 0.48, ("laplace", 300): 0.92, ("laplace", 1000): 0.94, ("laplace", 3000): 0.96, ("uniform", 100): 0.8, ("uniform", 300): 1.0, ("uniform", 1000): 1.0, ("uniform", 3000): 0.98}.items():
        assert s[key]["exact"] == pytest.approx(exact, abs=0.08), key
    assert s[("laplace", 100)]["correct"] == pytest.approx(0.82, abs=0.06) and s[("laplace", 100)]["missing"] == pytest.approx(0.38, abs=0.2)
    text = ev.verdict_gauss_experiment(r)[1]
    assert "Exzess-Kurtosis 0,00: 53 % richtig gerichtet (Zufall), 0,75: 92 %, 3,00: 100 %" in text and "Mit n = 100 findet LiNGAM bei Laplace-Rauschen in 48 %" in text and "mit n = 3 000 in 96 %" in text


def test_experiment_3_icp(full_icp):
    r = full_icp
    cum = {x["envs"]: x for x in r["cumulative"]}
    assert r["reps"] == 50 and r["n_e"] == 1000 and [x["name"] for x in r["cumulative"]] == list(K.ENV_NAMES[:5])
    assert cum[1]["empty"] == 1.0 and cum[2]["empty"] == 1.0 and cum[1]["exact"] == 0.0 and cum[3]["empty"] == pytest.approx(0.9, abs=TOL) and cum[3]["X"] == pytest.approx(0.1, abs=TOL)       # Konstruktion: ohne Eingriffe keine Unterscheidung
    assert cum[4]["W"] >= 0.95 and cum[4]["Z"] <= 0.1 and cum[4]["X"] <= 0.15 and cum[5]["exact"] == pytest.approx(0.74, abs=0.08) and cum[5]["Z"] == pytest.approx(0.76, abs=0.08) and cum[5]["X"] >= 0.95 and cum[5]["W"] >= 0.95
    assert all(x["coverage"] >= 0.9 for x in r["cumulative"] + r["sample"] + r["alpha"] + r["kinds"])                  # Garantie: höchstens α = 0,05 Irrtum (50 Läufe: Spielraum für Zufall)
    smp = {x["ne"]: x for x in r["sample"]}
    assert smp[100]["empty"] == pytest.approx(0.48, abs=0.1) and smp[100]["exact"] <= 0.1 and smp[300]["exact"] == pytest.approx(0.18, abs=0.1) and smp[300]["X"] == pytest.approx(0.5, abs=0.12) and smp[1000]["exact"] == pytest.approx(0.74, abs=0.08) and smp[3000]["exact"] == pytest.approx(0.96, abs=0.06)
    al = {x["alpha"]: x for x in r["alpha"]}
    assert al[0.01]["exact"] == pytest.approx(0.64, abs=0.1) and al[0.01]["coverage"] >= 0.96 and al[0.05]["exact"] == pytest.approx(0.7, abs=0.1) and al[0.01]["exact"] < al[0.05]["exact"] + 0.04
    kd = {x["kind"]: x for x in r["kinds"]}
    assert kd["gauss"]["exact"] == pytest.approx(0.74, abs=0.08) and kd["laplace"]["exact"] == pytest.approx(0.76, abs=0.08) and abs(kd["gauss"]["exact"] - kd["laplace"]["exact"]) <= 0.12
    text = ev.verdict_icp_experiment(r)[1]
    assert "Mit nur Beobachtungen gibt ICP die leere Menge (100 % der Läufe)" in text and "mit 5 Umgebungen (n_e = 1 000) findet es die drei Eltern genau in" in text and "α = 0,01 gibt " in text and "der Unterschied liegt im Rahmen der Zufallsstreuung" in text and "Zufallsstreuung bei 50 Läufen etwa ±3 Punkte" in text


def test_experiment_4_limits(full_limits):
    r = full_limits
    h, y, p = r["hidden"], r["y_intervention"], r["prediction"]
    assert r["reps"] == 30 and h["main"] == ("S → X", "S → Y", "W → Y", "X → Y", "Y → C") and h["share"] >= 0.95
    assert y["empty"] == 1.0 and y["coverage"] == 1.0                                                                  # Konstruktion der Methode: Verschiebung von Y macht jede Menge nicht invariant
    assert r["accepted_with_c"] == pytest.approx(0.52, abs=0.06) and r["accepted_count"] == pytest.approx(4.0, abs=0.4)
    assert p["wahre Eltern"]["mean"] == pytest.approx(1.0, abs=0.02) and p["ohne Beschwerden"]["mean"] == pytest.approx(1.0, abs=0.02) and p["alle Größen"]["mean"] == pytest.approx(1.81, abs=0.04) and p["nur Mittel"]["mean"] == pytest.approx(3.72, abs=0.08)
    assert p["ICP-Schnittmenge"]["mean"] > 5.0
    text = ev.verdict_limits(r)[1]
    assert "Ist die Nachfrage verborgen, gibt LiNGAM in 100 % der Läufe dieselbe Ausgabe (S → X, S → Y, W → Y, X → Y, Y → C)" in text and "gibt ICP in 100 % der Läufe die leere Menge" in text and "wahre Eltern 1,00, alle Größen mit den Beschwerden 1,81" in text
