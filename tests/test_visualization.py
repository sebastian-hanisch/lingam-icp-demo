"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import pytest

import lci_constants as K
import lci_evaluation as ev
import lci_visualization as V


def _arrows(fig):
    return [ann for ann in fig.layout.annotations if ann.showarrow]


def test_true_diagram_draws_every_arrow_and_marks_x_and_y():
    fig = V.build_true_diagram()
    assert len(_arrows(fig)) == 6
    nodes = {t.text[0]: t for t in fig.data if t.mode == "markers+text"}
    assert set(nodes) == set(K.NAMES) and nodes["X"].marker.line.color == K.COLOR_GOOD and nodes["Y"].marker.line.color == K.COLOR_TRUTH


def test_learned_diagram_is_green_when_everything_is_right():
    fig = V.build_learned_diagram(ev.analyse_lingam(ev.LingamSettings()))
    assert {ann.arrowcolor for ann in _arrows(fig)} == {K.COLOR_GOOD} and len(_arrows(fig)) == 6 and {t.name for t in fig.data if t.name} == {"richtig"}


def test_learned_diagram_shows_wrong_edges_for_gaussian_noise():
    a = ev.analyse_lingam(ev.LingamSettings(kind="Gauß"))
    fig = V.build_learned_diagram(a)
    names = {t.name for t in fig.data if t.name}
    assert "überzählig" in names and ("umgekehrt" in names or "fehlt" in names)
    assert len(_arrows(fig)) == len([e for e in a.status if e[2] != "fehlt"])


def test_edge_and_method_tables_by_hand():
    a = ev.analyse_lingam(ev.LingamSettings())
    t = V.build_edge_table(a)
    assert list(t.columns) == ["Kante", "Urteil", "Gewicht (nach der Regression)"] and len(t) == 6 and set(t["Urteil"]) == {"richtig"}
    row = t[t["Kante"] == "S → Z"].iloc[0]
    assert float(row["Gewicht (nach der Regression)"].replace("+", "").replace(",", ".").replace("−", "-")) == pytest.approx(0.8, abs=0.1)
    V.style_edge_table(V.build_edge_table(ev.analyse_lingam(ev.LingamSettings(kind="Gauß"))))
    m = V.build_methods_table(a)
    assert list(m["Verfahren"]) == list(ev.METHODS) and m.iloc[0]["richtig gerichtet"] == "6 von 6" and m.iloc[1]["ungerichtet"] == 2 and m.iloc[1]["SHD"] == "1,0"
    s = V.build_lingam_summary_table(a)
    assert len(s) == 4 and s.iloc[0]["Wert"] == "Laplace, 1,00" and s.iloc[1]["Wert"] == "3,00" and s.iloc[3]["Wert"] == "1000" and " → " in s.iloc[2]["Wert"]


def test_pair_chart_has_two_panels_and_the_right_titles():
    a = ev.analyse_lingam(ev.LingamSettings())
    fig = V.build_pair_chart(a)
    assert len(fig.data) == 2 and len(fig.data[0].x) == 400 and len(fig.layout.annotations) == 2 and "Nachfrage → Faktor (wahr)" in fig.layout.annotations[0].text and "Faktor → Nachfrage" in fig.layout.annotations[1].text
    assert ev.de(a.pair["a → b"]["dependence"], 2) in fig.layout.annotations[0].text and ev.de(a.pair["b → a"]["dependence"], 2) in fig.layout.annotations[1].text


def test_icp_chart_and_tables():
    b = ev.analyse_icp(ev.IcpSettings())
    fig = V.build_icp_chart(b)
    assert list(fig.data[0].x) == ["S", "Z", "X", "W", "C"] and list(fig.data[0].marker.color) == [K.COLOR_NEUTRAL, K.COLOR_GOOD, K.COLOR_GOOD, K.COLOR_GOOD, K.COLOR_NEUTRAL]
    assert list(fig.data[0].text) == ["", "★ identifiziert", "★ identifiziert", "★ identifiziert", ""] and fig.data[0].y[1] == 100.0
    t = V.build_icp_table(b)
    assert list(t.columns) == ["Menge S", "p-Wert der Invarianz", "angenommen"] and len(t) == 12 and t.iloc[0]["angenommen"] == "ja"
    s = V.build_icp_summary_table(b)
    assert s.iloc[3]["Wert"] == "{Z, X, W}" and s.iloc[4]["Wert"] == "{Z, X, W}" and s.iloc[2]["Wert"] == str(len(b.result.accepted)) and "beobachtet" in s.iloc[0]["Wert"]
    assert V.build_icp_summary_table(ev.analyse_icp(ev.IcpSettings(envs=())))["Wert"].iloc[0] == "keine"


def test_experiment_figures_and_tables_build(small_lingam, small_gauss, small_icp, small_limits):
    assert V.build_lingam_chart(small_lingam) is not None and len(V.build_lingam_table(small_lingam)) == 9 + 2
    assert V.build_transition_chart(small_gauss) is not None and len(V.build_transition_table(small_gauss)) == 3 and len(V.build_sample_table(small_gauss)) == 4
    assert V.build_icp_cumulative_chart(small_icp) is not None and [len(t) for t in V.build_icp_tables(small_icp)] == [5, 2, 2, 2]
    tables = V.build_limits_tables(small_limits)
    assert len(tables[0]) == 3 and len(tables[1]) == 5


def test_lingam_chart_groups_three_methods_by_noise(small_lingam):
    fig = V.build_lingam_chart(small_lingam)
    assert [t.name for t in fig.data] == ["LiNGAM roh", "LiNGAM standardisiert", "PC"] and list(fig.data[0].x) == ["Laplace", "gleichverteilt", "Gauß"] and fig.layout.barmode == "group"
    assert fig.data[0].y[0] == pytest.approx(100 * small_lingam["kinds"]["laplace"]["LiNGAM roh"]["exact"])


def test_transition_chart_marks_chance_and_icp_chart_has_four_lines(small_gauss, small_icp):
    fig = V.build_transition_chart(small_gauss)
    assert [t.name for t in fig.data] == ["Kanten richtig gerichtet", "genauer DAG"] and fig.layout.shapes[0].y0 == 50 and list(fig.data[0].x) == pytest.approx([0.0, 0.75, 3.0])
    icp = V.build_icp_cumulative_chart(small_icp)
    assert len(icp.data) == 4 and list(icp.data[0].x) == [1, 2, 3, 4, 5]
