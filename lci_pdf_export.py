"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import lci_constants as K
from lci_evaluation import METHODS, IcpAnalysis, LingamAnalysis, de, names_of, verdict_icp, verdict_lingam

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "α": "alpha", "σ": "sigma", "²": "2", "≤": "<=", "≥": ">=", "⊆": "in", "★": "*", "½": "1/2"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: LingamAnalysis, b: IcpAnalysis) -> bytes:
    s, si = a.settings, b.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Über die Äquivalenzklasse hinaus: LiNGAM und ICP"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Welt der Linie mit sechs Größen (S, Z, X, Y, W, C), Rauschen: {s.kind}, Anteil t = {de(s.t, 2)}; LiNGAM mit {s.n} Beobachtungen, ICP mit {si.ne} je Umgebung, alpha = {de(si.alpha, 2)}, "
                            f"Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "LiNGAM", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for x, y, status in a.status:
        pdf.multi_cell(0, 4.6, _s(f"Kante {K.NAMES[x]} -> {K.NAMES[y]}: {status}."), new_x="LMARGIN", new_y="NEXT")
    for m in METHODS:
        c = a.compare[m]
        pdf.multi_cell(0, 4.6, _s(f"{m}: {c['correct']} von {c['n_true']} richtig gerichtet, {c['reversed']} umgekehrt, {c['undirected']} ungerichtet, {c['missing']} fehlend, {c['extra']} überzählig (SHD {de(c['shd'], 1)})."),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_lingam(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _s("Invariante Kausalvorhersage für den Servicegrad"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.6, _s(f"Umgebungen: {', '.join(si.envs) or 'keine'}. Identifizierte Eltern: {names_of(b.identified)}; wahre Eltern: {names_of(K.TRUE_PARENTS_OF_Y)}; "
                            f"angenommene Mengen: {len(b.result.accepted)} von {len(b.table)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_icp(b)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: lingam-icp-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
