"""Konstanten: Welt mit sechs Größen, Regler-Grenzen, Umgebungen, feste Experimente, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35

# Welt: S → Z → X → Y, Z → Y, W → Y, Y → C (Einheitsrauschen, Pfeilstärken wie in den Stücken davor, neu: Beschwerden C als Wirkung des Servicegrads)
S, Z, X, Y, W, C_ = range(6)
NAMES = ("S", "Z", "X", "Y", "W", "C")
LONG = ("Saison", "Nachfrage", "Faktor", "Servicegrad", "Wetter", "Beschwerden")
COEF = {(S, Z): 0.8, (Z, X): 0.8, (X, Y): 0.6, (Z, Y): 0.6, (W, Y): 0.6, (Y, C_): 0.8}
TRUE_EDGES = frozenset(COEF)
TRUE_PARENTS_OF_Y = frozenset({Z, X, W})

KINDS = ("Laplace", "gleichverteilt", "Gauß")
KIND_KEYS = {"Laplace": "laplace", "gleichverteilt": "uniform", "Gauß": "gauss"}
DEFAULT_KIND = KINDS[0]
T_MIN, T_MAX, T_STEP, DEFAULT_T = 0.0, 1.0, 0.05, 1.0              # Anteil der gewählten Verteilung am Rauschen: √(1 − t) Gauß + √t Verteilung
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 3000, 100, 1000             # Beobachtungen für LiNGAM
NE_MIN, NE_MAX, NE_STEP, DEFAULT_NE = 100, 3000, 100, 1000          # Beobachtungen je Umgebung für ICP
ALPHAS = (0.01, 0.05, 0.1)
DEFAULT_ALPHA = 0.05
TARGET = Y

# Umgebungen (Eingriffe): s_shift Mittel von S, do_x X unabhängig von Z (N(0, 2²)), w_shift Mittel von W, z_scale Streuung des Rauschens von Z, y_shift Verschiebung von Y (verletzt die Annahme)
ENV_NAMES = ("beobachtet", "Saison verschoben", "Faktor zugeteilt", "Wetter verschoben", "Nachfrage streut", "Eingriff in den Servicegrad")
ENV_SPECS = {
    "beobachtet": {},
    "Saison verschoben": {"s_shift": 2.0},
    "Faktor zugeteilt": {"do_x": True},
    "Wetter verschoben": {"w_shift": 2.0},
    "Nachfrage streut": {"z_scale": 2.0},
    "Eingriff in den Servicegrad": {"y_shift": 2.0},
}
ENV_HELP = {
    "beobachtet": "Die Welt, wie sie ist.",
    "Saison verschoben": "Die Saison liegt im Mittel um 2 höher.",
    "Faktor zugeteilt": "Der Faktor wird unabhängig von der Nachfrage zugeteilt (N(0, 2²)): der Pfeil Nachfrage → Faktor ist durchtrennt.",
    "Wetter verschoben": "Das Wetter liegt im Mittel um 2 höher.",
    "Nachfrage streut": "Das Rauschen der Nachfrage hat die doppelte Streuung.",
    "Eingriff in den Servicegrad": "Der Servicegrad selbst wird um 2 verschoben: Das verletzt die Annahme von ICP (keine Eingriffe in die Zielgröße).",
}
DEFAULT_ENVS = ENV_NAMES[:5]

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1050000
EXP1_REPS, EXP1_NOTEARS_REPS, EXP1_N = 50, 12, 1000                     # Experiment 1: LiNGAM, PC (50), NOTEARS (12 Läufe, nur Laplace)
EXP2_TS, EXP2_REPS, EXP2_NS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0), 50, (100, 300, 1000, 3000)      # Experiment 2: Übergang zur Gauß-Form und Stichprobe
EXP3_REPS, EXP3_NE, EXP3_NES, EXP3_ALPHAS = 50, 1000, (100, 300, 1000, 3000), (0.01, 0.05)       # Experiment 3: ICP
EXP4_REPS, EXP4_TRAIN_ENVS = 30, 4                                                                # Experiment 4: Grenzen
EXP4_TEST_ENV = {"s_shift": 4.0, "w_shift": -4.0}

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(kind=DEFAULT_KIND, t=DEFAULT_T, n=DEFAULT_N, ne=DEFAULT_NE, alpha=DEFAULT_ALPHA, envs=DEFAULT_ENVS, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Laplace-Rauschen (Voreinstellung)": preset(),
    "Gleichverteiltes Rauschen": preset(kind="gleichverteilt"),
    "Gauß-Rauschen": preset(kind="Gauß"),
    "Fast Gauß": preset(t=0.25),
    "Wenige Umgebungen": preset(envs=ENV_NAMES[:4]),
    "Kleine Umgebungen": preset(ne=300),
    "Eingriff in den Servicegrad": preset(envs=ENV_NAMES),
}
PRESET_HELP = {
    "Laplace-Rauschen (Voreinstellung)": "Die Welt der Linie mit spitzem Rauschen und fünf Umgebungen: LiNGAM findet den ganzen DAG, auch die Richtungen, die PC offen lässt; ICP findet die Eltern des Servicegrads.",
    "Gleichverteiltes Rauschen": "Flaches Rauschen: auch nicht Gauß'sch, LiNGAM findet den DAG.",
    "Gauß-Rauschen": "Gauß'sches Rauschen: LiNGAM kann die Richtungen nicht mehr bestimmen (eine Auswahl wie ein Münzwurf, mit überzähligen Kanten); ICP ändert sich nicht.",
    "Fast Gauß": "Nur ein Viertel des Rauschens ist Laplace (Exzess-Kurtosis 0,19): Das reicht für LiNGAM nicht, die Richtungen sind kaum besser als Zufall.",
    "Wenige Umgebungen": "Ohne die Umgebung „Nachfrage streut“ sind die Nachfrage und der Faktor nicht zu sehen: ICP findet nur das Wetter.",
    "Kleine Umgebungen": "Nur 300 Beobachtungen je Umgebung: Die Invarianztests sind zu schwach, ICP nennt nur einen Teil der Eltern.",
    "Eingriff in den Servicegrad": "Zusätzlich wird der Servicegrad selbst verschoben: Keine Menge ist mehr invariant, ICP gibt die leere Menge zurück.",
}
