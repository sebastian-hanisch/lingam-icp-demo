"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import lci_constants as K
import lci_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("t_slider", "n_slider", "ne_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["LiNGAM: richtig gerichtet"] == "6 von 6" and m["PC: richtig gerichtet"] == "4 von 6" and m["NOTEARS roh: richtig gerichtet"] == "6 von 6" and m["Identifizierte Eltern"] == "{Z, X, W}" and m["Wahre Eltern"] == "{Z, X, W}"
    assert m["Ergebnis ⊆ wahre Eltern"] == "ja" and int(m["NOTEARS standardisiert: richtig gerichtet"].split(" ")[0]) <= 3
    assert at.radio(key="kind_radio").value == K.DEFAULT_KIND and at.slider(key="t_slider").value == K.DEFAULT_T and at.slider(key="n_slider").value == K.DEFAULT_N and at.slider(key="ne_slider").value == K.DEFAULT_NE
    assert at.select_slider(key="alpha_slider").value == K.DEFAULT_ALPHA and at.number_input(key="seed_input").value == K.DEFAULT_SEED and all(at.checkbox(key=P.ENV_KEYS[n]).value == (n in K.DEFAULT_ENVS) for n in K.ENV_NAMES)
    assert any("LiNGAM trifft das wahre Diagramm genau" in s.value for s in at.success) and any("ICP findet genau die Eltern" in s.value for s in at.success)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # die Experimente laufen in eigenen Tests
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in K.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.radio(key="kind_radio").value == preset["kind"] and at.slider(key="t_slider").value == preset["t"] and at.slider(key="n_slider").value == preset["n"] and at.slider(key="ne_slider").value == preset["ne"]
        assert at.select_slider(key="alpha_slider").value == preset["alpha"] and at.number_input(key="seed_input").value == preset["seed"]
        assert {n for n in K.ENV_NAMES if at.checkbox(key=P.ENV_KEYS[n]).value} == set(preset["envs"])


def test_gaussian_preset_shows_the_warning_and_icp_is_unchanged():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Gauß-Rauschen")].click().run(timeout=TIMEOUT)
    assert not at.exception and any("(fast) Gauß'sch" in w.value for w in at.warning) and _metrics(at)["Identifizierte Eltern"] == "{Z, X, W}"
    assert int(_metrics(at)["LiNGAM: richtig gerichtet"].split(" ")[0]) < 6


def test_y_intervention_preset_gives_the_empty_set():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Eingriff in den Servicegrad")].click().run(timeout=TIMEOUT)
    assert not at.exception and _metrics(at)["Identifizierte Eltern"] == "{}" and any("selbst verschoben" in w.value for w in at.warning)


def test_unchecking_every_environment_does_not_raise():
    at = _fresh_app()
    for n in K.ENV_NAMES:
        at.checkbox(key=P.ENV_KEYS[n]).set_value(False)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(w.value.startswith("Keine Umgebung gewählt") for w in at.warning) and _metrics(at)["Identifizierte Eltern"] == "{}"


def test_choice_widgets_work():
    at = _fresh_app()
    at.radio(key="kind_radio").set_value("gleichverteilt").run(timeout=TIMEOUT)
    at.select_slider(key="alpha_slider").set_value(0.1).run(timeout=TIMEOUT)
    assert not at.exception and at.radio(key="kind_radio").value == "gleichverteilt" and at.select_slider(key="alpha_slider").value == 0.1


def test_every_experiment_runs_on_click():
    at = _fresh_app()
    assert "exp1_ready" not in at.session_state
    for key, text in (("exp1_run", "findet LiNGAM in"), ("exp2_run", "Je mehr Rauschen Laplace statt Gauß ist"), ("exp3_run", "Mit nur Beobachtungen gibt ICP die leere Menge"), ("exp4_run", "Ist die Nachfrage verborgen, gibt LiNGAM")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
        assert any(text in i.value for i in at.info), key


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "50"
    at.query_params["t"] = "9"
    at.query_params["k"] = "2"
    at.query_params["a"] = "0"
    at.query_params["e"] = "9"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == K.N_MIN and at.slider(key="t_slider").value == K.T_MAX and at.radio(key="kind_radio").value == "Gauß" and at.select_slider(key="alpha_slider").value == 0.01
    assert {n for n in K.ENV_NAMES if at.checkbox(key=P.ENV_KEYS[n]).value} == {"beobachtet", "Wetter verschoben"} and at.number_input(key="seed_input").value == K.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    assert at.query_params["e"] in ("31", ["31"])
    at.slider(key="t_slider").set_value(0.5).run(timeout=TIMEOUT)
    assert at.query_params["t"] in ("0.5", ["0.5"])
    at.checkbox(key=P.ENV_KEYS["Eingriff in den Servicegrad"]).set_value(True).run(timeout=TIMEOUT)
    assert at.query_params["e"] in ("63", ["63"])
    at.radio(key="kind_radio").set_value("Gauß").run(timeout=TIMEOUT)
    assert at.query_params["k"] in ("2", ["2"])
