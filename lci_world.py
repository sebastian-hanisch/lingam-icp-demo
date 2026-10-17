"""Die Welt mit sechs Größen (S, Z, X, Y, W, C), Rauschfamilien und Umgebungen, aus festen Strömen. Kein Streamlit."""

from __future__ import annotations

import math

import numpy as np

import lci_constants as K
from kausal_generator import SplitMix64


def noise(kind: str, rng: SplitMix64, size: int, t: float = 1.0) -> np.ndarray:
    """Rauschen mit Mittel 0 und Varianz 1: gauss, laplace, uniform; t mischt mit Gauß: √(1 − t) · Gauß + √t · Verteilung (Exzess-Kurtosis 3 t² bei Laplace, −1,2 t² bei gleichverteilt)."""
    if kind == "gauss":
        return rng.normal(size)
    if kind == "uniform":
        base = (rng.uniform(size) - 0.5) * math.sqrt(12.0)
    elif kind == "laplace":
        u = rng.uniform(size) - 0.5
        base = -np.sign(u) * np.log(1.0 - 2.0 * np.abs(u)) / math.sqrt(2.0)
    else:
        raise ValueError(kind)
    if t >= 1.0:
        return base
    return math.sqrt(1.0 - t) * rng.normal(size) + math.sqrt(t) * base


def simulate(n: int, rng: SplitMix64, kind: str = "laplace", t: float = 1.0, env: dict | None = None) -> np.ndarray:
    """n × 6 (S, Z, X, Y, W, C). Umgebung: s_shift, w_shift, z_scale, do_x, y_shift (siehe lci_constants.ENV_SPECS). Die Rauschspalten kommen nacheinander aus einem Strom."""
    env = env or {}
    e = np.column_stack([noise(kind, rng, n, t) for _ in range(6)])
    s = env.get("s_shift", 0.0) + e[:, K.S]
    z = K.COEF[(K.S, K.Z)] * s + env.get("z_scale", 1.0) * e[:, K.Z]
    x = 2.0 * e[:, K.X] if env.get("do_x") else K.COEF[(K.Z, K.X)] * z + e[:, K.X]
    w = env.get("w_shift", 0.0) + e[:, K.W]
    y = K.COEF[(K.X, K.Y)] * x + K.COEF[(K.Z, K.Y)] * z + K.COEF[(K.W, K.Y)] * w + env.get("y_shift", 0.0) + e[:, K.Y]
    c = K.COEF[(K.Y, K.C_)] * y + e[:, K.C_]
    return np.column_stack([s, z, x, y, w, c])


def make_environments(names, n_e: int, seed: int, kind: str = "gauss", t: float = 1.0):
    """Daten und Umgebungsnummern: je Umgebung n_e Beobachtungen aus eigenem Strom (seed + 1000 · Nummer); names sind Schlüssel von ENV_SPECS oder Wörterbücher."""
    datas, labels = [], []
    for k, name in enumerate(names):
        spec = K.ENV_SPECS[name] if isinstance(name, str) else name
        datas.append(simulate(n_e, SplitMix64(seed + 1000 * k), kind, t, spec))
        labels.append(np.full(n_e, k))
    return np.vstack(datas), np.concatenate(labels)


def excess_kurtosis(kind: str, t: float) -> float:
    """Exzess-Kurtosis des Rauschens (Kumulanten addieren sich): Laplace 3 t², gleichverteilt −1,2 t², Gauß 0."""
    return {"laplace": 3.0, "uniform": -1.2, "gauss": 0.0}[kind] * t * t
