"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lci_constants as K
from lci_evaluation import METHODS, IcpAnalysis, LingamAnalysis, de, names_of, pct, signed

POS = {K.S: (0.0, 1.0), K.Z: (0.32, 1.0), K.W: (0.95, 1.0), K.X: (0.32, 0.0), K.Y: (0.66, 0.5), K.C_: (1.0, 0.0)}
STATUS_COLORS = {"richtig": K.COLOR_GOOD, "umgekehrt": K.COLOR_TRUTH, "überzählig": K.COLOR_BAD, "fehlt": K.COLOR_NEUTRAL}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segment(p0, p1, r: float = 0.1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    return (p0[0] + ux * r, p0[1] + uy * r), (p1[0] - ux * r, p1[1] - uy * r)


def _arrow(fig: go.Figure, a: int, b: int, color: str, width: float = 3):
    s, e = _segment(POS[a], POS[b])
    fig.add_annotation(x=e[0], y=e[1], ax=s[0], ay=s[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def _nodes(fig: go.Figure):
    outlines = {K.X: K.COLOR_GOOD, K.Y: K.COLOR_TRUTH}
    for v, (x, y) in POS.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[K.NAMES[v]], textposition="middle center", textfont=dict(size=12),
                                 marker=dict(size=58, color="rgba(47,107,101,0.12)", line=dict(width=2, color=outlines.get(v, "#1b1b1b"))), hoverinfo="skip", showlegend=False))


def _frame(fig: go.Figure, height: int = 330, **layout) -> go.Figure:
    fig.update_xaxes(visible=False, range=[-0.2, 1.2])
    fig.update_yaxes(visible=False, range=[-0.3, 1.3])
    return _fixed(fig, height, **layout)


def build_true_diagram() -> go.Figure:
    fig = go.Figure()
    for a, b in sorted(K.TRUE_EDGES):
        _arrow(fig, a, b, "#6b6b6b")
    _nodes(fig)
    return _frame(fig, showlegend=False)


def build_learned_diagram(a: LingamAnalysis) -> go.Figure:
    """Der von LiNGAM gelernte Graph nach Urteil gefärbt (richtig, umgekehrt, überzählig); fehlende wahre Kanten gestrichelt grau."""
    fig = go.Figure()
    present = set()
    for x, y, st in a.status:
        present.add(st)
        if st == "fehlt":
            s, e = _segment(POS[x], POS[y], 0.1)
            fig.add_trace(go.Scatter(x=[s[0], e[0]], y=[s[1], e[1]], mode="lines", line=dict(color=K.COLOR_NEUTRAL, width=2, dash="dash"), hoverinfo="skip", showlegend=False))
        else:
            _arrow(fig, x, y, STATUS_COLORS[st], 4 if st != "richtig" else 3)
    _nodes(fig)
    for st in ("richtig", "umgekehrt", "überzählig", "fehlt"):
        if st in present:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=STATUS_COLORS[st], width=4, dash="dash" if st == "fehlt" else "solid"), name=st))
    return _frame(fig, showlegend=True)


def build_edge_table(a: LingamAnalysis) -> pd.DataFrame:
    return pd.DataFrame([{"Kante": f"{K.NAMES[x]} → {K.NAMES[y]}", "Urteil": st, "Gewicht (nach der Regression)": signed(a.weights[x, y], 2) if st != "fehlt" else ""} for x, y, st in a.status])


def style_edge_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Urteil"] != "richtig" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_methods_table(a: LingamAnalysis) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren": m, "richtig gerichtet": f"{a.compare[m]['correct']} von {a.compare[m]['n_true']}", "umgekehrt": a.compare[m]["reversed"], "ungerichtet": a.compare[m]["undirected"],
                          "fehlend": a.compare[m]["missing"], "überzählig": a.compare[m]["extra"], "SHD": de(a.compare[m]["shd"], 1)} for m in METHODS])


def build_lingam_summary_table(a: LingamAnalysis) -> pd.DataFrame:
    s = a.settings
    return pd.DataFrame([
        {"Größe": "Rauschen: Verteilung, Anteil t", "Wert": f"{s.kind}, {de(s.t, 2)}"},
        {"Größe": "Exzess-Kurtosis des Rauschens (Laplace 3 t², gleichverteilt −1,2 t², Gauß 0)", "Wert": de(a.excess, 2)},
        {"Größe": "Reihenfolge der Ursachen (LiNGAM, zuerst die exogenste)", "Wert": " → ".join(K.NAMES[v] for v in a.order)},
        {"Größe": "Beobachtungen", "Wert": str(s.n)},
    ])


def build_pair_chart(a: LingamAnalysis) -> go.Figure:
    """Nachfrage und Faktor in beiden Richtungen: Regressor gegen Residuum; in der wahren Richtung unabhängig, in der anderen nicht (bei nicht Gauß'schem Rauschen)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Nachfrage → Faktor (wahr): Quadrate-Korrelation {de(a.pair['a → b']['dependence'], 2)}", f"Faktor → Nachfrage: {de(a.pair['b → a']['dependence'], 2)}"))
    for col, key in ((1, "a → b"), (2, "b → a")):
        u, r = a.pair[key]["regressor"][:400], a.pair[key]["residual"][:400]
        fig.add_trace(go.Scatter(x=u, y=r, mode="markers", marker=dict(size=4, color=K.COLOR_GOOD if col == 1 else K.COLOR_TRUTH, opacity=0.5), showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(title_text="Regressor" , row=1, col=col)
        fig.update_yaxes(title_text="Residuum" if col == 1 else None, row=1, col=col)
    return _fixed(fig, 330, showlegend=False)


# ----------------------------------------------------------------------------- ICP

def build_icp_chart(b: IcpAnalysis) -> go.Figure:
    labels = [K.NAMES[j] for j in b.result.others]
    colors = [K.COLOR_GOOD if j in K.TRUE_PARENTS_OF_Y else K.COLOR_NEUTRAL for j in b.result.others]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * b.share[j] for j in b.result.others], marker_color=colors, name="Anteil der angenommenen Mengen, die die Größe enthalten",
                         text=["★ identifiziert" if j in b.identified else "" for j in b.result.others], textposition="outside"))
    fig.update_yaxes(title="angenommene Mengen mit dieser Größe (%)", range=[0, 118])
    fig.update_xaxes(title="Größe (grün: wahres Elternteil des Servicegrads)")
    return _fixed(fig, 330, showlegend=False)


def build_icp_table(b: IcpAnalysis, limit: int = 12) -> pd.DataFrame:
    rows = [{"Menge S": names_of(sub), "p-Wert der Invarianz": de(p, 3), "angenommen": "ja" if acc else "nein"} for sub, p, acc in b.table[:limit]]
    return pd.DataFrame(rows)


def build_icp_summary_table(b: IcpAnalysis) -> pd.DataFrame:
    s = b.settings
    return pd.DataFrame([
        {"Größe": "Umgebungen", "Wert": ", ".join(s.envs) if s.envs else "keine"},
        {"Größe": "Beobachtungen je Umgebung, Niveau α", "Wert": f"{s.ne}, {de(s.alpha, 2)}"},
        {"Größe": "Angenommene Mengen (von 32)", "Wert": str(len(b.result.accepted))},
        {"Größe": "Identifizierte Eltern (Schnittmenge)", "Wert": names_of(b.identified)},
        {"Größe": "Wahre Eltern", "Wert": names_of(K.TRUE_PARENTS_OF_Y)},
    ])


# ----------------------------------------------------------------------------- Experimente

def build_lingam_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    kinds = ["laplace", "uniform", "gauss"]
    labels = {"laplace": "Laplace", "uniform": "gleichverteilt", "gauss": "Gauß"}
    for method, color in (("LiNGAM roh", K.COLOR_GOOD), ("LiNGAM standardisiert", K.COLOR_CONTROL), ("PC", K.COLOR_NEUTRAL)):
        fig.add_trace(go.Bar(x=[labels[k] for k in kinds], y=[100 * res["kinds"][k][method]["exact"] for k in kinds], name=method, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Läufe mit genau richtigem DAG (%)", range=[0, 104])
    fig.update_xaxes(title="Rauschverteilung")
    return _fixed(fig, 360)


def build_lingam_table(res: dict) -> pd.DataFrame:
    labels = {"laplace": "Laplace", "uniform": "gleichverteilt", "gauss": "Gauß"}
    rows = []
    for k, ms in res["kinds"].items():
        for m, r in ms.items():
            rows.append({"Rauschen": labels[k], "Verfahren": m, "genau": pct(r["exact"]), "richtig gerichtet": pct(r["correct"], 1), "umgekehrt": pct(r["reversed"], 1), "ungerichtet": pct(r["undirected"], 1),
                         "überzählig je Lauf": de(r["extra"], 2), "SHD": de(r["shd"], 2)})
    for m, r in res["notears"].items():
        rows.append({"Rauschen": f"Laplace ({res['notears_reps']} Läufe)", "Verfahren": m, "genau": pct(r["exact"]), "richtig gerichtet": pct(r["correct"], 1), "umgekehrt": pct(r["reversed"], 1), "ungerichtet": pct(r["undirected"], 1),
                     "überzählig je Lauf": de(r["extra"], 2), "SHD": de(r["shd"], 2)})
    return pd.DataFrame(rows)


def build_transition_chart(res: dict) -> go.Figure:
    rows = res["transition"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["excess"] for r in rows], y=[100 * r["correct"] for r in rows], mode="lines+markers", name="Kanten richtig gerichtet", line=dict(color=K.COLOR_GOOD, width=3)))
    fig.add_trace(go.Scatter(x=[r["excess"] for r in rows], y=[100 * r["exact"] for r in rows], mode="lines+markers", name="genauer DAG", line=dict(color=K.COLOR_CONTROL, width=3, dash="dash")))
    fig.add_hline(y=50, line=dict(color="#888888", width=1, dash="dot"), annotation_text="Zufall (50 %)", annotation_position="right")
    fig.update_xaxes(title="Exzess-Kurtosis des Rauschens (Gauß 0, Laplace 3)")
    fig.update_yaxes(title="Läufe bzw. Kanten (%)", range=[-2, 104])
    return _fixed(fig, 340)


def build_transition_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Anteil Laplace t": de(r["t"], 2), "Exzess-Kurtosis": de(r["excess"], 2), "richtig gerichtet": pct(r["correct"], 1), "umgekehrt": pct(r["reversed"], 1), "genau": pct(r["exact"]),
                          "überzählig je Lauf": de(r["extra"], 2)} for r in res["transition"]])


def build_sample_table(res: dict) -> pd.DataFrame:
    labels = {"laplace": "Laplace", "uniform": "gleichverteilt"}
    return pd.DataFrame([{"Rauschen": labels[r["kind"]], "n": r["n"], "genau": pct(r["exact"]), "richtig gerichtet": pct(r["correct"], 1), "SHD": de(r["shd"], 2)} for r in res["sample"]])


def build_icp_cumulative_chart(res: dict) -> go.Figure:
    rows = res["cumulative"]
    fig = go.Figure()
    for key, label, color in (("Z", "Nachfrage Z", K.COLOR_TRUTH), ("X", "Faktor X", K.COLOR_GOOD), ("W", "Wetter W", K.COLOR_CONTROL)):
        fig.add_trace(go.Scatter(x=[r["envs"] for r in rows], y=[100 * r[key] for r in rows], mode="lines+markers", name=f"{label} identifiziert", line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(x=[r["envs"] for r in rows], y=[100 * r["false"] for r in rows], mode="lines+markers", name="falsches Elternteil genannt", line=dict(color=K.COLOR_BAD, width=2, dash="dash")))
    fig.update_xaxes(title="Zahl der Umgebungen (kumulativ: beobachtet, Saison, Faktor, Wetter, Nachfrage)", dtick=1)
    fig.update_yaxes(title="Läufe (%)", range=[-2, 104])
    return _fixed(fig, 360)


def build_icp_tables(res: dict) -> list:
    cum = pd.DataFrame([{"Umgebungen": r["envs"], "bis": r["name"], "alle drei Eltern": pct(r["exact"]), "Z": pct(r["Z"]), "X": pct(r["X"]), "W": pct(r["W"]), "Ergebnis ⊆ Eltern": pct(r["coverage"]), "leere Menge": pct(r["empty"])}
                        for r in res["cumulative"]])
    smp = pd.DataFrame([{"n je Umgebung": r["ne"], "alle drei Eltern": pct(r["exact"]), "Z": pct(r["Z"]), "X": pct(r["X"]), "W": pct(r["W"]), "Ergebnis ⊆ Eltern": pct(r["coverage"]), "leere Menge": pct(r["empty"])} for r in res["sample"]])
    alpha = pd.DataFrame([{"α": de(r["alpha"], 2), "alle drei Eltern": pct(r["exact"]), "Ergebnis ⊆ Eltern": pct(r["coverage"]), "falsches Elternteil": pct(r["false"])} for r in res["alpha"]])
    kinds = pd.DataFrame([{"Rauschen": r["kind"], "alle drei Eltern": pct(r["exact"]), "Ergebnis ⊆ Eltern": pct(r["coverage"])} for r in res["kinds"]])
    return [cum, smp, alpha, kinds]


def build_limits_tables(res: dict) -> list:
    h, y = res["hidden"], res["y_intervention"]
    rows = [{"Fall": "LiNGAM, Nachfrage verborgen", "Befund": f"in {pct(h['share'])} der {h['reps']} Läufe dieselbe Ausgabe: {', '.join(h['main'])}"},
            {"Fall": "ICP, Eingriff in den Servicegrad", "Befund": f"leere Menge in {pct(y['empty'])} der Läufe, Ergebnis ⊆ Eltern in {pct(y['coverage'])}"},
            {"Fall": "ICP ohne diesen Eingriff: angenommene Mengen mit den Beschwerden", "Befund": f"{pct(res['accepted_with_c'])} der angenommenen Mengen enthalten C; im Mittel {de(res['accepted_count'], 1)} von 32 Mengen werden angenommen"}]
    pred = pd.DataFrame([{"Vorhersage von Y": k, "Fehler in der neuen Umgebung": de(v["mean"], 3), "Streuung": de(v["sd"], 3)} for k, v in res["prediction"].items()])
    return [pd.DataFrame(rows), pred]
