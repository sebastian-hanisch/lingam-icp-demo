"""Auswertung: LiNGAM mit den Vergleichszeilen PC und NOTEARS, ICP für den Servicegrad, ein Paar zum Anschauen, die vier Experimente über feste Ströme und die Verdikte. Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import lci_constants as K
import lci_icp as I
import lci_lingam as G
import lci_world as W
import nts_baselines as B
import nts_notears as N
import stl_pc as P
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


def _n(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def names_of(cols) -> str:
    cols = sorted(cols)
    return "{" + ", ".join(K.NAMES[c] for c in cols) + "}" if cols else "{}"


@dataclass(frozen=True)
class LingamSettings:
    kind: str = K.DEFAULT_KIND
    t: float = K.DEFAULT_T
    n: int = K.DEFAULT_N
    seed: int = K.DEFAULT_SEED

    @property
    def key(self) -> str:
        return K.KIND_KEYS[self.kind]


@dataclass(frozen=True)
class IcpSettings:
    kind: str = K.DEFAULT_KIND
    t: float = K.DEFAULT_T
    ne: int = K.DEFAULT_NE
    alpha: float = K.DEFAULT_ALPHA
    envs: tuple = K.DEFAULT_ENVS
    seed: int = K.DEFAULT_SEED

    @property
    def key(self) -> str:
        return K.KIND_KEYS[self.kind]


@dataclass(frozen=True)
class Settings:
    kind: str = K.DEFAULT_KIND
    t: float = K.DEFAULT_T
    n: int = K.DEFAULT_N
    ne: int = K.DEFAULT_NE
    alpha: float = K.DEFAULT_ALPHA
    envs: tuple = K.DEFAULT_ENVS
    seed: int = K.DEFAULT_SEED

    @property
    def lingam(self) -> LingamSettings:
        return LingamSettings(self.kind, self.t, self.n, self.seed)

    @property
    def icp(self) -> IcpSettings:
        return IcpSettings(self.kind, self.t, self.ne, self.alpha, self.envs, self.seed)


def edge_status(learned: set, truth) -> list:
    out = []
    for a, b in sorted(learned):
        out.append((a, b, "richtig" if (a, b) in truth else "umgekehrt" if (b, a) in truth else "überzählig"))
    skel = {frozenset(e) for e in learned}
    for a, b in sorted(truth):
        if frozenset((a, b)) not in skel:
            out.append((a, b, "fehlt"))
    return out


METHODS = ("LiNGAM", "PC", "NOTEARS roh", "NOTEARS standardisiert")


# ----------------------------------------------------------------------------- Analyse: LiNGAM

@dataclass
class LingamAnalysis:
    settings: LingamSettings
    excess: float                 # Exzess-Kurtosis des Rauschens
    order: list
    edges: set
    status: list
    weights: np.ndarray
    compare: dict                 # je Verfahren der Vergleich mit dem wahren Diagramm
    status_other: dict            # Kantenurteile von NOTEARS roh
    pair: dict                    # Paar Nachfrage → Faktor, beide Richtungen
    pc: tuple


def analyse_lingam(s: LingamSettings) -> LingamAnalysis:
    data = W.simulate(s.n, SplitMix64(s.seed), s.key, s.t)
    res = G.direct_lingam(data)
    truth = set(K.TRUE_EDGES)
    pc = P.pc(6, P.fisher_z_test(np.corrcoef(data, rowvar=False), len(data), 0.01)).graph
    nt_raw = N.threshold_dag(N.notears(data).w, 0.3)
    nt_std = N.threshold_dag(N.notears(B.standardize(data)).w, 0.3)
    compare = {"LiNGAM": B.compare(res.edges, set(), truth), "PC": B.compare(set(pc[0]), set(pc[1]), truth), "NOTEARS roh": B.compare(nt_raw, set(), truth), "NOTEARS standardisiert": B.compare(nt_std, set(), truth)}
    return LingamAnalysis(s, W.excess_kurtosis(s.key, s.t), res.order, res.edges, edge_status(res.edges, truth), res.weights, compare, {"NOTEARS roh": edge_status(nt_raw, truth)},
                          G.pair_dependence(data[:, K.Z], data[:, K.X]), pc)


# ----------------------------------------------------------------------------- Analyse: ICP

@dataclass
class IcpAnalysis:
    settings: IcpSettings
    result: I.IcpResult
    identified: set
    covered: bool                 # identifizierte Menge ⊆ wahre Eltern
    exact: bool
    empty: bool
    table: list                   # (Menge, p-Wert, angenommen), nach p absteigend
    share: dict                   # Anteil der angenommenen Mengen, die die Größe enthalten
    y_intervention: bool


def analyse_icp(s: IcpSettings) -> IcpAnalysis:
    data, env = W.make_environments(list(s.envs) or [K.ENV_NAMES[0]], s.ne, s.seed + 17, s.key, s.t)        # ohne gewählte Umgebung: nur Beobachtungen
    res = I.icp(data, env, K.TARGET, s.alpha)
    table = sorted(((sub, p, sub in res.accepted) for sub, p in res.p_values.items()), key=lambda r: -r[1])
    share = {j: (float(np.mean([j in sub for sub in res.accepted])) if res.accepted else 0.0) for j in res.others}
    return IcpAnalysis(s, res, res.intersection, res.intersection <= K.TRUE_PARENTS_OF_Y, res.intersection == K.TRUE_PARENTS_OF_Y, not res.intersection, table, share, "Eingriff in den Servicegrad" in s.envs)


# ----------------------------------------------------------------------------- Experimente

def _aggregate(rows: list) -> dict:
    n_true = sum(r["n_true"] for r in rows)
    return {"correct": sum(r["correct"] for r in rows) / n_true, "reversed": sum(r["reversed"] for r in rows) / n_true, "undirected": sum(r["undirected"] for r in rows) / n_true,
            "missing": float(np.mean([r["missing"] for r in rows])), "extra": float(np.mean([r["extra"] for r in rows])), "shd": float(np.mean([r["shd"] for r in rows])),
            "exact": float(np.mean([r["shd"] == 0 for r in rows]))}


def lingam_experiment(reps: int = K.EXP1_REPS, notears_reps: int = K.EXP1_NOTEARS_REPS, n: int = K.EXP1_N) -> dict:
    """LiNGAM roh und standardisiert und PC bei drei Rauschfamilien; NOTEARS roh und standardisiert nur bei Laplace-Rauschen (weniger Läufe)."""
    truth = set(K.TRUE_EDGES)
    rows = {kind: {"LiNGAM roh": [], "LiNGAM standardisiert": [], "PC": []} for kind in ("laplace", "uniform", "gauss")}
    notears = {"NOTEARS roh": [], "NOTEARS standardisiert": []}
    for i, kind in enumerate(rows):
        for r in range(reps):
            data = W.simulate(n, SplitMix64(K.SEED_BASE + 100000 * i + r), kind)
            rows[kind]["LiNGAM roh"].append(B.compare(G.direct_lingam(data).edges, set(), truth))
            rows[kind]["LiNGAM standardisiert"].append(B.compare(G.direct_lingam(B.standardize(data)).edges, set(), truth))
            g = P.pc(6, P.fisher_z_test(np.corrcoef(data, rowvar=False), n, 0.01)).graph
            rows[kind]["PC"].append(B.compare(set(g[0]), set(g[1]), truth))
            if kind == "laplace" and r < notears_reps:
                notears["NOTEARS roh"].append(B.compare(N.threshold_dag(N.notears(data).w, 0.3), set(), truth))
                notears["NOTEARS standardisiert"].append(B.compare(N.threshold_dag(N.notears(B.standardize(data)).w, 0.3), set(), truth))
    return {"reps": reps, "notears_reps": notears_reps, "n": n, "kinds": {k: {m: _aggregate(v) for m, v in ms.items()} for k, ms in rows.items()}, "notears": {m: _aggregate(v) for m, v in notears.items()}}


def gauss_experiment(ts=K.EXP2_TS, ns=K.EXP2_NS, reps: int = K.EXP2_REPS) -> dict:
    """Übergang zur Gauß-Form (Laplace-Mischung, n = 1 000) und Stichprobe (Laplace und gleichverteilt, volle Verteilung)."""
    truth = set(K.TRUE_EDGES)
    transition = []
    for i, t in enumerate(ts):
        rows = [B.compare(G.direct_lingam(W.simulate(1000, SplitMix64(K.SEED_BASE + 1000000 + 10000 * i + r), "laplace", t)).edges, set(), truth) for r in range(reps)]
        transition.append({"t": t, "excess": W.excess_kurtosis("laplace", t), **_aggregate(rows)})
    sample = []
    for k, kind in enumerate(("laplace", "uniform")):
        for j, n in enumerate(ns):
            rows = [B.compare(G.direct_lingam(W.simulate(n, SplitMix64(K.SEED_BASE + 2000000 + 100000 * k + 10000 * j + r), kind)).edges, set(), truth) for r in range(reps)]
            sample.append({"kind": kind, "n": n, **_aggregate(rows)})
    return {"reps": reps, "transition": transition, "sample": sample}


def _icp_stats(results: list) -> dict:
    pa = K.TRUE_PARENTS_OF_Y
    return {"coverage": float(np.mean([o <= pa for o in results])), "exact": float(np.mean([o == pa for o in results])), "Z": float(np.mean([K.Z in o for o in results])), "X": float(np.mean([K.X in o for o in results])),
            "W": float(np.mean([K.W in o for o in results])), "false": float(np.mean([bool(o - pa) for o in results])), "empty": float(np.mean([len(o) == 0 for o in results])), "size": float(np.mean([len(o) for o in results]))}


def _run_icp(envs, n_e: int, reps: int, kind: str, alpha: float, seed: int, t: float = 1.0) -> list:
    out = []
    for r in range(reps):
        data, env = W.make_environments(envs, n_e, seed + 17 * r, kind, t)
        out.append(I.icp(data, env, K.TARGET, alpha).intersection)
    return out


def icp_experiment(reps: int = K.EXP3_REPS, n_e: int = K.EXP3_NE, nes=K.EXP3_NES, alphas=K.EXP3_ALPHAS) -> dict:
    """ICP: Umgebungen kumulativ (Gauß), Stichprobe je Umgebung, α und Rauschfamilie."""
    cumulative = [{"envs": k, "name": K.ENV_NAMES[k - 1], **_icp_stats(_run_icp(K.ENV_NAMES[:k], n_e, reps, "gauss", 0.05, K.SEED_BASE + 3000000 + 100000 * k))} for k in range(1, 6)]
    sample = [{"ne": ne, **_icp_stats(_run_icp(K.DEFAULT_ENVS, ne, reps, "gauss", 0.05, K.SEED_BASE + 4000000 + 100000 * j))} for j, ne in enumerate(nes)]
    alpha_rows = [{"alpha": a, **_icp_stats(_run_icp(K.DEFAULT_ENVS, n_e, reps, "gauss", a, K.SEED_BASE + 5000000))} for a in alphas]
    kinds = [{"kind": kind, **_icp_stats(_run_icp(K.DEFAULT_ENVS, n_e, reps, kind, 0.05, K.SEED_BASE + 3000000 + 500000 * (k + 1)))} for k, kind in enumerate(("gauss", "laplace"))]
    return {"reps": reps, "n_e": n_e, "cumulative": cumulative, "sample": sample, "alpha": alpha_rows, "kinds": kinds}


def limits_experiment(reps: int = K.EXP4_REPS) -> dict:
    """Wenn die Annahme fehlt: verborgene Nachfrage bei LiNGAM, Eingriff in den Servicegrad bei ICP, Vorhersage von Y in einer neuen Umgebung."""
    keep = [K.S, K.X, K.Y, K.W, K.C_]
    names = {i: K.NAMES[v] for i, v in enumerate(keep)}
    outputs: dict = {}
    for r in range(reps):
        edges = G.direct_lingam(W.simulate(1000, SplitMix64(K.SEED_BASE + 6000000 + r), "laplace")[:, keep]).edges
        key = tuple(sorted(f"{names[a]} → {names[b]}" for a, b in edges))
        outputs[key] = outputs.get(key, 0) + 1
    main, count = max(outputs.items(), key=lambda kv: kv[1])
    hidden = {"main": main, "share": count / reps, "reps": reps}
    with_y = _icp_stats(_run_icp(K.ENV_NAMES, 1000, reps, "gauss", 0.05, K.SEED_BASE + 7000000))
    accepted_c = []
    for r in range(reps):
        data, env = W.make_environments(K.DEFAULT_ENVS, 1000, K.SEED_BASE + 8000000 + 17 * r, "gauss")
        res = I.icp(data, env, K.TARGET)
        accepted_c.append((float(np.mean([K.C_ in sub for sub in res.accepted])) if res.accepted else 0.0, len(res.accepted)))
    errors: dict = {k: [] for k in ("alle Größen", "ohne Beschwerden", "wahre Eltern", "ICP-Schnittmenge", "nur Mittel")}
    for r in range(reps):
        data, env = W.make_environments(K.ENV_NAMES[:K.EXP4_TRAIN_ENVS], 1000, K.SEED_BASE + 9000000 + 17 * r, "gauss")
        icp_set = I.icp(data, env, K.TARGET).intersection
        test = W.simulate(2000, SplitMix64(K.SEED_BASE + 9900000 + r), "gauss", 1.0, K.EXP4_TEST_ENV)
        test[:, K.C_] = SplitMix64(K.SEED_BASE + 9990000 + r).normal(2000)          # im neuen Eingriff hängen die Beschwerden nicht mehr am Servicegrad

        def fit(cols):
            z = np.column_stack([np.ones(len(data))] + ([data[:, cols]] if cols else []))
            beta, *_ = np.linalg.lstsq(z, data[:, K.Y], rcond=None)
            zt = np.column_stack([np.ones(len(test))] + ([test[:, cols]] if cols else []))
            return float(np.mean((test[:, K.Y] - zt @ beta) ** 2))

        errors["alle Größen"].append(fit([K.S, K.Z, K.X, K.W, K.C_]))
        errors["ohne Beschwerden"].append(fit([K.S, K.Z, K.X, K.W]))
        errors["wahre Eltern"].append(fit([K.Z, K.X, K.W]))
        errors["ICP-Schnittmenge"].append(fit(sorted(icp_set)))
        errors["nur Mittel"].append(fit([]))
    return {"reps": reps, "hidden": hidden, "y_intervention": with_y, "accepted_with_c": float(np.mean([a for a, _ in accepted_c])), "accepted_count": float(np.mean([b for _, b in accepted_c])),
            "prediction": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in errors.items()}}


# ----------------------------------------------------------------------------- Verdikte

def verdict_lingam(a: LingamAnalysis) -> tuple:
    c = a.compare
    nt = c["LiNGAM"]["n_true"]
    head = (f"LiNGAM richtet {c['LiNGAM']['correct']} von {nt} wahren Kanten richtig (SHD {de(c['LiNGAM']['shd'], 1)}); PC richtet {c['PC']['correct']} und lässt {c['PC']['undirected']} ungerichtet (SHD {de(c['PC']['shd'], 1)}); "
            f"NOTEARS roh {c['NOTEARS roh']['correct']}, standardisiert {c['NOTEARS standardisiert']['correct']} (SHD {de(c['NOTEARS roh']['shd'], 1)} und {de(c['NOTEARS standardisiert']['shd'], 1)}). "
            f"Exzess-Kurtosis des Rauschens: {de(a.excess, 2)}.")
    if c["LiNGAM"]["shd"] == 0:
        return "success", head + " LiNGAM trifft das wahre Diagramm genau, aus Beobachtungen allein, weil das Rauschen nicht Gauß'sch ist."
    wrong = "; ".join(f"{K.NAMES[x]} → {K.NAMES[y]} {u}" for x, y, u in a.status if u != "richtig")
    if abs(a.excess) < 0.3:
        return "warning", head + f" Das Rauschen ist (fast) Gauß'sch: Die Richtungen sind nicht bestimmt, die Reihenfolge ist eine Auswahl wie ein Münzwurf ({wrong})."
    return "warning", head + f" Abweichungen: {wrong}."


def verdict_pair(a: LingamAnalysis) -> tuple:
    f, b = a.pair["a → b"]["dependence"], a.pair["b → a"]["dependence"]
    if abs(a.excess) < 0.3:
        return "info", (f"Das Rauschen ist (fast) Gauß'sch: Die Abhängigkeit zwischen Regressor und Residuum ist in beiden Richtungen nahe null ({de(f, 2)} und {de(b, 2)}); die Richtung ist nicht erkennbar.")
    return "info", (f"Nachfrage → Faktor: Regressor und Residuum sind in der wahren Richtung unabhängig (Korrelation der Quadrate {de(f, 2)}), in der umgekehrten nicht ({de(b, 2)}). Das ist, was LiNGAM nutzt; bei Gauß-Rauschen wären beide null.")


def verdict_icp(a: IcpAnalysis) -> tuple:
    n_env = len(a.settings.envs)
    if not a.settings.envs:
        return "warning", "Keine Umgebung gewählt. ICP braucht Umgebungen mit Eingriffen; mit den Beobachtungen allein bleibt die Menge leer."
    if a.y_intervention and a.empty:
        return "warning", f"In einer der {n_env} Umgebungen wird der Servicegrad selbst verschoben: Keine Menge ist invariant, ICP gibt die leere Menge zurück. Das ist vorsichtig, aber auch ohne Aussage."
    if a.empty:
        return "warning", f"Leere Menge: Mit {n_env} Umgebung(en) und {a.settings.ne} Beobachtungen je Umgebung ist keine Teilmenge der Eltern von den übrigen zu unterscheiden. ICP behauptet dann nichts."
    if a.exact:
        return "success", f"ICP findet genau die Eltern des Servicegrads: {names_of(a.identified)} (angenommene Mengen: {len(a.result.accepted)} von {len(a.table)}). Bei erfüllten Annahmen nennt ICP mit Wahrscheinlichkeit mindestens {pct(1 - a.settings.alpha)} nur Eltern."
    if not a.covered:
        return "error", f"ICP nennt {names_of(a.identified)}, darunter Größen, die keine Eltern sind: ein Fehler, der mit Wahrscheinlichkeit höchstens {pct(a.settings.alpha)} auftritt."
    missing = names_of(K.TRUE_PARENTS_OF_Y - a.identified)
    return "info", f"ICP nennt {names_of(a.identified)}, aber nicht {missing}: Mit diesen Umgebungen sind die fehlenden Eltern nicht von Größen zu unterscheiden, die den Servicegrad nur vorhersagen. Die Schnittmenge ist eine Untergrenze, kein Irrtum."


def verdict_lingam_experiment(res: dict) -> tuple:
    k = res["kinds"]
    nt = res["notears"]
    return "info", (f"Bei Laplace-Rauschen (n = {_n(res['n'])}, {res['reps']} Läufe) findet LiNGAM in {pct(k['laplace']['LiNGAM roh']['exact'])} der Läufe den genauen DAG, bei gleichverteiltem in {pct(k['uniform']['LiNGAM roh']['exact'])}, bei Gauß-Rauschen "
                    f"in {pct(k['gauss']['LiNGAM roh']['exact'])} (richtig gerichtet {pct(k['gauss']['LiNGAM roh']['correct'])}, überzählige Kanten je Lauf {de(k['gauss']['LiNGAM roh']['extra'], 2)}). Standardisiert ändert sich nichts "
                    f"({pct(k['laplace']['LiNGAM standardisiert']['exact'])} bei Laplace). PC lässt {pct(k['laplace']['PC']['undirected'])} der Kanten ungerichtet; NOTEARS richtet roh {pct(nt['NOTEARS roh']['correct'])} richtig, standardisiert "
                    f"{pct(nt['NOTEARS standardisiert']['correct'])} ({res['notears_reps']} Läufe).")


def verdict_gauss_experiment(res: dict) -> tuple:
    tr = {r["t"]: r for r in res["transition"]}
    first, mid, last = res["transition"][0], tr[0.5], res["transition"][-1]
    sample = {(r["kind"], r["n"]): r for r in res["sample"]}
    small, large = min(r["n"] for r in res["sample"]), max(r["n"] for r in res["sample"])
    return "info", (f"Je mehr Rauschen Laplace statt Gauß ist, desto besser die Richtungen: Exzess-Kurtosis {de(first['excess'], 2)}: {pct(first['correct'])} richtig gerichtet (Zufall), {de(mid['excess'], 2)}: {pct(mid['correct'])}, "
                    f"{de(last['excess'], 2)}: {pct(last['correct'])}; der Übergang liegt dazwischen, nicht bei null. Mit n = {small} findet LiNGAM bei Laplace-Rauschen in {pct(sample[('laplace', small)]['exact'])} der Läufe den genauen DAG, "
                    f"mit n = {_n(large)} in {pct(sample[('laplace', large)]['exact'])}.")


def verdict_icp_experiment(res: dict) -> tuple:
    cum, smp = res["cumulative"], {r["ne"]: r for r in res["sample"]}
    al = {r["alpha"]: r for r in res["alpha"]}
    kd = {r["kind"]: r for r in res["kinds"]}
    return "info", (f"Mit nur Beobachtungen gibt ICP die leere Menge ({pct(cum[0]['empty'])} der Läufe); mit {len(cum)} Umgebungen (n_e = {_n(res['n_e'])}) findet es die drei Eltern genau in {pct(cum[-1]['exact'])}, "
                    f"mit n_e = {min(smp)} in {pct(smp[min(smp)]['exact'])}, mit n_e = {_n(max(smp))} in {pct(smp[max(smp)]['exact'])}. Ein falsches Elternteil nennt es bei α = 0,05 in höchstens {pct(max(r['false'] for r in res['cumulative'] + res['sample'] + res['kinds'] + [a for a in res['alpha'] if a['alpha'] == 0.05]))} "
                    f"der Läufe (Zufallsstreuung bei {res['reps']} Läufen etwa ±3 Punkte). α = 0,01 gibt {pct(al[0.01]['exact'])} genau (α = 0,05: {pct(al[0.05]['exact'])}; der Unterschied liegt im Rahmen der Zufallsstreuung). Gauß- und Laplace-Rauschen geben dasselbe ({pct(kd['gauss']['exact'])} und {pct(kd['laplace']['exact'])}).")


def verdict_limits(res: dict) -> tuple:
    h, y, p = res["hidden"], res["y_intervention"], res["prediction"]
    return "info", (f"Ist die Nachfrage verborgen, gibt LiNGAM in {pct(h['share'])} der Läufe dieselbe Ausgabe ({', '.join(h['main'])}): zwei falsche Kanten, alles gerichtet, keine Warnung. "
                    f"Wird der Servicegrad in einer Umgebung selbst verschoben, gibt ICP in {pct(y['empty'])} der Läufe die leere Menge. Vorhersage in einer neuen Umgebung (Rest 1,00): wahre Eltern {de(p['wahre Eltern']['mean'], 2)}, "
                    f"alle Größen mit den Beschwerden {de(p['alle Größen']['mean'], 2)}, die ICP-Schnittmenge allein {de(p['ICP-Schnittmenge']['mean'], 2)} (sie enthält nur einen Teil der Eltern).")
