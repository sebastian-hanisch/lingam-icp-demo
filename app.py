"""Über die Äquivalenzklasse hinaus: LiNGAM und invariante Kausalvorhersage - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünfzehntes Stück der Linie Kausale Diagramme und Adjustierung. Zwei Wege, die Richtung zu finden, die Beobachtungen allein nicht geben: die Gestalt des Rauschens (LiNGAM: nicht Gauß'sches Rauschen macht den ganzen DAG
bestimmbar) und Umgebungen mit Eingriffen (invariante Kausalvorhersage: die Eltern des Servicegrads sind eine Menge von Größen, gegeben derer seine Verteilung in allen Umgebungen gleich bleibt).
Kern: lci_lingam.py, lci_icp.py, lci_world.py, lci_evaluation.py; PC und NOTEARS als Vergleich: stl_pc.py, nts_notears.py (wortgleich aus Stück 13 und 14).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lci_constants as K
from lci_evaluation import (
    Settings,
    analyse_icp,
    analyse_lingam,
    de,
    gauss_experiment,
    icp_experiment,
    limits_experiment,
    lingam_experiment,
    names_of,
    pct,
    verdict_gauss_experiment,
    verdict_icp,
    verdict_icp_experiment,
    verdict_limits,
    verdict_lingam,
    verdict_lingam_experiment,
    verdict_pair,
)
from lci_pdf_export import build_pdf
from lci_presets import ENV_KEYS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from lci_visualization import (
    build_edge_table,
    build_icp_chart,
    build_icp_cumulative_chart,
    build_icp_summary_table,
    build_icp_table,
    build_icp_tables,
    build_learned_diagram,
    build_limits_tables,
    build_lingam_chart,
    build_lingam_summary_table,
    build_lingam_table,
    build_methods_table,
    build_pair_chart,
    build_sample_table,
    build_transition_chart,
    build_transition_table,
    build_true_diagram,
    style_edge_table,
)

st.set_page_config(page_title="Über die Äquivalenzklasse hinaus – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner="LiNGAM, PC und NOTEARS rechnen …")
def _lingam(settings):
    return analyse_lingam(settings)


@st.cache_data(show_spinner=False)
def _icp(settings):
    return analyse_icp(settings)


@st.cache_data(show_spinner=SPINNER)
def _exp1():
    return lingam_experiment()


@st.cache_data(show_spinner=SPINNER)
def _exp2():
    return gauss_experiment()


@st.cache_data(show_spinner=SPINNER)
def _exp3():
    return icp_experiment()


@st.cache_data(show_spinner=SPINNER)
def _exp4():
    return limits_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def experiment_button(label, key):
    if st.button(label, key=f"{key}_run"):
        st.session_state[f"{key}_ready"] = True
    return st.session_state.get(f"{key}_ready", False)


st.title("🔀 Über die Äquivalenzklasse hinaus")
st.markdown(
    """
**Wie kommt man zur Richtung, wenn Beobachtungen sie nicht geben?** Dies ist das fünfzehnte Stück der Linie **Kausale Diagramme und Adjustierung**. PC (Stück 13) endet bei der Äquivalenzklasse, NOTEARS (Stück 14) richtet Kanten nur,
solange die Skala die Reihenfolge verrät. Hier zwei Wege, die wirklich etwas Neues hineinstecken: **LiNGAM** nutzt, dass das Rauschen **nicht Gauß'sch** ist (dann ist, bei linearem Modell ohne verborgene Ursachen, der ganze DAG aus Beobachtungen bestimmbar, unabhängig von der Skala),
**invariante Kausalvorhersage (ICP)** nutzt **Umgebungen mit Eingriffen** (die Eltern des Servicegrads sind eine Menge von Größen, gegeben derer seine Verteilung in allen Umgebungen gleich bleibt; ICP nennt die Schnittmenge aller solchen Mengen). Beide scheitern leise, wenn ihre Annahme fehlt.

Weiter unten: das **Paar zum Anschauen**, vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man das Ergebnis", expanded=True):
    st.markdown(
        """
1. **Die Welt** (links): die der Linie mit sechs Größen: Saison S → Nachfrage Z → Faktor X → Servicegrad Y, dazu Z → Y, Wetter W → Y und neu die **Beschwerden C** als Wirkung des Servicegrads.
2. **LiNGAM** (rechts): **grün** richtig gerichtet, **orange** umgekehrt, **rot** überzählig, **grau gestrichelt** fehlend. Bei Gauß-Rauschen ist die Reihenfolge nicht bestimmt, das Ergebnis eine Auswahl.
3. **Vergleich auf denselben Daten:** PC lässt Kanten ungerichtet, NOTEARS richtet roh nach der Varianz und verliert die Richtungen nach einer Standardisierung, LiNGAM ist skalenfrei.
4. **ICP** (Abschnitt darunter): Für jede Menge von Größen wird getestet, ob die Verteilung des Servicegrads gegeben diese Größen in allen Umgebungen gleich ist. Die **Schnittmenge** aller angenommenen Mengen sind Eltern, bei gültigen Tests mit Wahrscheinlichkeit höchstens α ein Irrtum.
5. **Umgebungen** (Seitenleiste): Ohne Eingriffe gibt ICP die leere Menge; ein Eingriff in den Servicegrad selbst verletzt die Annahme.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
names = list(K.PRESETS)
for chunk in (names[:4], names[4:]):
    cols = st.columns(4)
    for col, name in zip(cols, chunk):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=K.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Rauschen**")
    kind = st.radio("Verteilung des Rauschens", K.KINDS, key="kind_radio", help="Laplace: spitz (Exzess-Kurtosis 3), gleichverteilt: flach (−1,2), Gauß: 0. Nur nicht Gauß'sches Rauschen macht die Richtungen für LiNGAM bestimmbar.")
    t = st.slider("Anteil t der gewählten Verteilung", *bounds("t_slider"), step=K.T_STEP, key="t_slider", help="Das Rauschen ist √(1 − t) · Gauß + √t · gewählte Verteilung (Varianz 1). t = 0 ist reines Gauß-Rauschen.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen für LiNGAM", *bounds("n_slider"), step=K.N_STEP, key="n_slider", help="Zeilen der Datenmatrix (nur beobachtet).")
    ne = st.slider("Beobachtungen je Umgebung (ICP)", *bounds("ne_slider"), step=K.NE_STEP, key="ne_slider", help="Je gewählte Umgebung so viele Standorte.")
    alpha = st.select_slider("Niveau α der Invarianztests", options=K.ALPHAS, key="alpha_slider", help="ICP nennt mit Wahrscheinlichkeit höchstens α eine Größe, die kein Elternteil ist. Kleineres α: weniger Treffer, höhere Sicherheit.")
    st.markdown("**Umgebungen für ICP**")
    for name, key in ENV_KEYS.items():
        st.checkbox(name, key=key, help=K.ENV_HELP[name])
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

envs = tuple(name for name, key in ENV_KEYS.items() if st.session_state[key])
sync_query_params({"kind_radio": kind, "t_slider": round(float(t), 2), "n_slider": int(n), "ne_slider": int(ne), "alpha_slider": alpha, "seed_input": int(seed)}, envs)

settings = Settings(kind, round(float(t), 2), int(n), int(ne), float(alpha), envs, int(seed))
a = _lingam(settings.lingam)
b = _icp(settings.icp)

# --- Ergebnis: LiNGAM ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 LiNGAM: der DAG aus Beobachtungen")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
cmp_ = a.compare
nt = cmp_["LiNGAM"]["n_true"]
c1.metric("LiNGAM: richtig gerichtet", f"{cmp_['LiNGAM']['correct']} von {nt}", delta=f"{cmp_['LiNGAM']['reversed']} umgekehrt, SHD {de(cmp_['LiNGAM']['shd'], 1)}", delta_color="off", help="Wahre Kanten, die LiNGAM mit richtiger Richtung findet.")
c2.metric("PC: richtig gerichtet", f"{cmp_['PC']['correct']} von {nt}", delta=f"{cmp_['PC']['undirected']} ungerichtet, SHD {de(cmp_['PC']['shd'], 1)}", delta_color="off", help="PC (stabil, Fisher-z, α = 0,01); ungerichtete Kanten zählen im SHD halb.")
c3.metric("NOTEARS roh: richtig gerichtet", f"{cmp_['NOTEARS roh']['correct']} von {nt}", delta=f"{cmp_['NOTEARS roh']['reversed']} umgekehrt, SHD {de(cmp_['NOTEARS roh']['shd'], 1)}", delta_color="off", help="Auf den Rohdaten richtet die Varianzordnung.")
c4.metric("NOTEARS standardisiert: richtig gerichtet", f"{cmp_['NOTEARS standardisiert']['correct']} von {nt}", delta=f"{cmp_['NOTEARS standardisiert']['reversed']} umgekehrt, SHD {de(cmp_['NOTEARS standardisiert']['shd'], 1)}", delta_color="off",
          help="Dieselben Daten, jede Spalte auf Streuung 1: LiNGAM ändert sich nicht, NOTEARS verliert die Richtungen.")

left, right = st.columns(2)
with left:
    st.plotly_chart(build_true_diagram(), width="stretch", key="true_chart")
    st.caption("Das wahre Diagramm. X (grün umrandet) ist der Faktor, Y (orange umrandet) der Servicegrad, C die Beschwerden.")
with right:
    st.plotly_chart(build_learned_diagram(a), width="stretch", key="learned_chart")
    st.caption("Was LiNGAM aus den Beobachtungen baut, gefärbt nach dem Urteil gegen das wahre Diagramm.")
verdict(verdict_lingam(a))
st.dataframe(style_edge_table(build_edge_table(a)), hide_index=True, width="stretch")
st.dataframe(build_methods_table(a), hide_index=True, width="stretch")
st.dataframe(build_lingam_summary_table(a), hide_index=True, width="stretch")

st.markdown("### 🔍 Ein Paar zum Anschauen: warum die Richtung erkennbar ist")
st.plotly_chart(build_pair_chart(a), width="stretch", key="pair_chart")
verdict(verdict_pair(a))
st.caption("Nachfrage und Faktor aus den Beobachtungen oben: links die Regression in der wahren Richtung (Residuum gegen Regressor), rechts die umgekehrte. Bei nicht Gauß'schem Rauschen sind die Residuen nur in der wahren Richtung unabhängig vom Regressor.")

st.markdown("---")

# --- Ergebnis: ICP --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🧪 ICP: die Eltern des Servicegrads aus Umgebungen")
(d1, d2), (d3, d4) = st.columns(2), st.columns(2)
d1.metric("Identifizierte Eltern", names_of(b.identified), help="Schnittmenge aller Mengen, mit denen die Verteilung des Servicegrads in allen Umgebungen gleich erscheint.")
d2.metric("Wahre Eltern", names_of(K.TRUE_PARENTS_OF_Y), help="Nachfrage, Faktor und Wetter wirken direkt auf den Servicegrad.")
d3.metric("Angenommene Mengen", f"{len(b.result.accepted)} von {len(b.table)}", help="Mengen der übrigen fünf Größen, deren Invarianz nicht verworfen wird (p > α).")
d4.metric("Ergebnis ⊆ wahre Eltern", "ja" if b.covered else "nein", delta=f"Umgebungen: {len(envs)}", delta_color="off", help="ICP garantiert bei gültigen Tests und ohne Eingriff in den Servicegrad: mit Wahrscheinlichkeit mindestens 1 − α liegt das Ergebnis in den wahren Eltern.")
st.plotly_chart(build_icp_chart(b), width="stretch", key="icp_chart")
verdict(verdict_icp(b))
st.dataframe(build_icp_table(b), hide_index=True, width="stretch")
st.dataframe(build_icp_summary_table(b), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a, b))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="lingam-icp.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Welche Verfahren verlassen die Klasse?")
st.caption(f"LiNGAM roh und standardisiert und PC bei drei Rauschverteilungen ({K.EXP1_REPS} feste Läufe, n = {K.EXP1_N}); NOTEARS roh und standardisiert bei Laplace-Rauschen ({K.EXP1_NOTEARS_REPS} feste Läufe).")
if experiment_button("▶ Experiment 1 rechnen (etwa 1 Minute)", "exp1"):
    res1 = _exp1()
    st.plotly_chart(build_lingam_chart(res1), width="stretch", key="lingam_chart")
    verdict(verdict_lingam_experiment(res1))
    st.dataframe(build_lingam_table(res1), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Wie nicht Gauß'sch muss das Rauschen sein?")
st.caption(f"Das Rauschen mischt Gauß und Laplace (t); {K.EXP2_REPS} feste Läufe je Zelle. Dazu die Stichprobe bei Laplace- und gleichverteiltem Rauschen.")
if experiment_button("▶ Experiment 2 rechnen (etwa 30 Sekunden)", "exp2"):
    res2 = _exp2()
    st.plotly_chart(build_transition_chart(res2), width="stretch", key="transition_chart")
    verdict(verdict_gauss_experiment(res2))
    st.dataframe(build_transition_table(res2), hide_index=True, width="stretch")
    st.dataframe(build_sample_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Wie viele Umgebungen braucht ICP?")
st.caption(f"{K.EXP3_REPS} feste Läufe je Zelle, Gauß-Rauschen (ICP hängt nicht davon ab). Umgebungen kumulativ, dann die Stichprobe je Umgebung, α und die Rauschverteilung.")
if experiment_button("▶ Experiment 3 rechnen (etwa 30 Sekunden)", "exp3"):
    res3 = _exp3()
    st.plotly_chart(build_icp_cumulative_chart(res3), width="stretch", key="icp_cumulative_chart")
    verdict(verdict_icp_experiment(res3))
    for table in build_icp_tables(res3):
        st.dataframe(table, hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Wenn die Annahme fehlt")
st.caption(f"{K.EXP4_REPS} feste Läufe: LiNGAM mit verborgener Nachfrage, ICP mit einem Eingriff in den Servicegrad, und die Vorhersage des Servicegrads in einer neuen Umgebung (Saison +4, Wetter −4, Beschwerden unabhängig).")
if experiment_button("▶ Experiment 4 rechnen (etwa 10 Sekunden)", "exp4"):
    res4 = _exp4()
    verdict(verdict_limits(res4))
    for table in build_limits_tables(res4):
        st.dataframe(table, hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Rauschen ist nicht Gauß'sch (LiNGAM)** | Bei Gauß-Rauschen ist die Reihenfolge nicht bestimmt: eine Auswahl, mit überzähligen Kanten; schon bei kleinem nicht-gaußschem Anteil (Exzess-Kurtosis unter etwa 0,75) kippt es (Experimente 1 und 2). | Wie falsch darf das Diagramm sein? |
| **Linear, alle Größen beobachtet (LiNGAM)** | Eine verborgene Ursache gibt einen sauber gerichteten Graphen mit falschen Kanten, ohne Warnung (Experiment 4). Nichtlineare additive Rauschmodelle sind nicht gebaut. | Wann geht es gar nicht? |
| **Bekannte Umgebungen ohne Eingriff in die Zielgröße (ICP)** | Wird der Servicegrad selbst verschoben, ist keine Menge invariant: leere Menge (Experiment 4). | Wie falsch darf das Diagramm sein? |
| **Genug Stärke und Stichprobe (ICP)** | Zu wenige oder zu schwache Umgebungen, zu kleine Stichprobe: ICP nennt nur einen Teil der Eltern oder nichts (Experiment 3). | Welches Verfahren, gemessen woran? |
| **Die Schnittmenge ist keine Vorhersage (ICP)** | Sie ist eine Untergrenze der Eltern; ein Modell nur mit ihr sagt den Servicegrad schlecht voraus (Experiment 4). | nicht gebaut |
| **Ein Zielknoten, bekannte Eingriffe** | Unbekannte Eingriffe und mehrere Ziele sind nicht gebaut. | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Lineares Strukturmodell $x = Bx + e$ mit unabhängigen Störungen $e_j$ (Varianz 1). **LiNGAM:** Sind höchstens eine der $e_j$ Gauß'sch, ist $B$ (bis auf die Skala) eindeutig bestimmt und die Reihenfolge der Variablen ablesbar; bei Gauß'schen $e_j$ nicht.
Die Verteilung des Rauschens hier: $\sqrt{1-t}\,\mathcal N(0,1) + \sqrt{t}\,L$ mit Exzess-Kurtosis $3t^2$ (Laplace) bzw. $-1{,}2\,t^2$ (gleichverteilt), denn Kumulanten addieren sich.

**DirectLiNGAM.** Für standardisierte $x_i, x_j$ mit Residuen $r_{i|j}$ (von $x_i$ nach Regression auf $x_j$) misst $\Delta H = H(x_j) + H(r_{i|j}) - H(x_i) - H(r_{j|i})$ (positiv: $x_i$ ist Ursache). Die exogenste Variable $i$ minimiert $\sum_{j \ne i} \min(0, \Delta H_{ij})^2$;
ihr Einfluss wird herausregressiert, der Schritt wiederholt. Die Entropie wird genähert durch $H(u) \approx \tfrac12(1 + \ln 2\pi) - k_1\,(E[\log\cosh u] - \gamma)^2 - k_2\,(E[u\,e^{-u^2/2}])^2$ (für $N(0,1)$ gleich $1{,}4189$).

**ICP.** Für eine Zielgröße $Y$ und jede Menge $S$ der übrigen Größen: $H_{0,S}$: die Verteilung von $Y$ gegeben $X_S$ ist in allen Umgebungen gleich. Geprüft über die Residuen der gepoolten Regression: gleicher Mittelwert (einfaktorielle Varianzanalyse, $F = \frac{SS_b/(k-1)}{SS_w/(n-k)}$)
und gleiche Streuung (Levene-Test auf $|r - \mathrm{Median}|$), Bonferroni über beide. Die Schnittmenge $\hat S = \bigcap \{S : H_{0,S}\ \text{nicht verworfen}\}$ erfüllt $P(\hat S \subseteq \mathrm{Pa}(Y)) \ge 1 - \alpha$, wenn die Umgebungen keine Eingriffe in $Y$ enthalten.
Die $F$-Verteilung wird über die unvollständige Betafunktion $I_x(a, b)$ ausgewertet: $P(F > f) = I_{d_2/(d_2 + d_1 f)}(d_2/2, d_1/2)$.

**Literatur.** S. Shimizu, P. Hoyer, A. Hyvärinen, A. Kerminen, *A linear non-Gaussian acyclic model for causal discovery*, JMLR 2006; S. Shimizu u. a., *DirectLiNGAM*, JMLR 2011; A. Hyvärinen, S. Smith, *Pairwise likelihood ratios for estimation of non-Gaussian structural equation models*, JMLR 2013;
J. Peters, P. Bühlmann, N. Meinshausen, *Causal inference by using invariant prediction*, JRSS-B 2016 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
