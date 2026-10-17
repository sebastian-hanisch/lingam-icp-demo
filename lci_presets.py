"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet;
es gibt weder KEPT-Regler noch berechnete Grenzen. Auswahlen (Rauschfamilie, α) stehen in der Adresszeile als Index, die sechs Umgebungen als Bitmaske (Parameter e)."""

import random
from dataclasses import dataclass

import streamlit as st

import lci_constants as K


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float
    choices: tuple | None = None


def _choice(param: str, choices: tuple, default) -> SettingSpec:
    return SettingSpec(param, int, default, 0, len(choices) - 1, 1, choices)


SETTING_SPECS = {
    "kind_radio": _choice("k", K.KINDS, K.DEFAULT_KIND),
    "t_slider": SettingSpec("t", float, K.DEFAULT_T, K.T_MIN, K.T_MAX, K.T_STEP),
    "n_slider": SettingSpec("n", int, K.DEFAULT_N, K.N_MIN, K.N_MAX, K.N_STEP),
    "ne_slider": SettingSpec("ne", int, K.DEFAULT_NE, K.NE_MIN, K.NE_MAX, K.NE_STEP),
    "alpha_slider": _choice("a", K.ALPHAS, K.DEFAULT_ALPHA),
    "seed_input": SettingSpec("seed", int, K.DEFAULT_SEED, 0, K.SEED_MAX, 1),
}
PRESET_KEYS = {"kind": "kind_radio", "t": "t_slider", "n": "n_slider", "ne": "ne_slider", "alpha": "alpha_slider", "seed": "seed_input"}
ENV_KEYS = {name: f"env_{i}" for i, name in enumerate(K.ENV_NAMES)}
ENV_PARAM = "e"


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen (bei einer Auswahl: auf einen gültigen Index)."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1 or spec.choices):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return spec.choices[int(value)] if spec.choices else value


def envs_to_mask(envs) -> int:
    return sum(1 << i for i, name in enumerate(K.ENV_NAMES) if name in envs)


def mask_to_envs(mask: int) -> tuple:
    mask = max(0, min((1 << len(K.ENV_NAMES)) - 1, int(mask)))
    return tuple(name for i, name in enumerate(K.ENV_NAMES) if mask >> i & 1)


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    for name, key in ENV_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = name in K.DEFAULT_ENVS


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    if ENV_PARAM in qp:
        try:
            chosen = mask_to_envs(int(float(qp[ENV_PARAM])))
        except (ValueError, TypeError):
            chosen = None
        if chosen is not None:
            for name, key in ENV_KEYS.items():
                st.session_state[key] = name in chosen
    st.session_state["permalink_loaded"] = True


def url_value(state_key: str, value):
    spec = SETTING_SPECS[state_key]
    return spec.choices.index(value) if spec.choices else value


def sync_query_params(values: dict, envs: tuple):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(url_value(state_key, value))
        st.query_params[ENV_PARAM] = str(envs_to_mask(envs))
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = K.PRESETS[name][key]
    for env_name, key in ENV_KEYS.items():
        st.session_state[key] = env_name in K.PRESETS[name]["envs"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, K.SEED_MAX)
