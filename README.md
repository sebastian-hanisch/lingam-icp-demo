# Über die Äquivalenzklasse hinaus: LiNGAM und invariante Kausalvorhersage – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lingam-icp-demo.streamlit.app/)**

Fünfzehntes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [notears-demo](https://github.com/sebastian-hanisch/notears-demo): PC (Stück 13) endet bei der Äquivalenzklasse, NOTEARS (Stück 14) richtet Kanten nur, solange die Skala die Reihenfolge verrät. Hier zwei Wege, die wirklich etwas Neues hineinstecken.

## Kernfrage

**LiNGAM** nutzt, dass das Rauschen **nicht Gauß'sch** ist: Dann ist (bei linearem Modell ohne verborgene Ursachen) der ganze DAG aus Beobachtungen bestimmbar, unabhängig von der Skala der Daten. **Invariante Kausalvorhersage (ICP)** nutzt **Umgebungen mit Eingriffen**: Die Eltern des Servicegrads sind eine Menge von Größen, gegeben derer seine Verteilung in allen Umgebungen gleich bleibt; ICP nennt die Schnittmenge aller solchen Mengen. Beide scheitern leise, wenn ihre Annahme fehlt: LiNGAM gibt einen gerichteten, aber falschen Graphen aus, ICP nennt keine Eltern (die leere Menge).

## Modell

- **Welt:** die der Linie mit **sechs** Größen: Saison S → Nachfrage Z → Faktor X → Servicegrad Y, dazu Z → Y, Wetter W → Y und neu die **Beschwerden C** als Wirkung des Servicegrads (Y → C 0,8); Pfeilstärken 0,8 / 0,8 / 0,6 / 0,6 / 0,6 / 0,8, Rauschen mit Varianz 1. Wahre Eltern von Y: {Z, X, W}.
- **Rauschen:** √(1 − t) · Gauß + √t · Verteilung (Laplace oder gleichverteilt), Varianz 1; Exzess-Kurtosis **3 t²** (Laplace) bzw. **−1,2 t²** (gleichverteilt); t = 0 ist reines Gauß-Rauschen.
- **Umgebungen** (Schalter): beobachtet; Saison verschoben (+2); Faktor zugeteilt (X unabhängig von Z, N(0, 2²)); Wetter verschoben (+2); Nachfrage streut (doppelte Streuung); dazu **Eingriff in den Servicegrad** (+2), der die Annahme von ICP verletzt.

## Methodik

- **DirectLiNGAM** (`lci_lingam.py`, von Grund auf in numpy): die exogenste Größe suchen (Summe der min(0, ΔH)² über die Paare, ΔH der Entropie-Näherung nach Hyvärinen/Smith 2013), deren Einfluss herausregressieren, wiederholen; dann Regression auf die Vorgänger, Kanten mit |t| < 3 gestrichen. Die Näherung ist für N(0, 1) auf die Gauß-Entropie **1,4189** geeicht; die Konstanten sind aus dem Gedächtnis, belegt durch das Verhalten.
- **ICP** (`lci_icp.py`): je Menge S der übrigen fünf Größen (32 Mengen) die gepoolte Regression von Y auf X_S; die Residuen müssen in allen Umgebungen gleichen Mittelwert (einfaktorielle Varianzanalyse) und gleiche Streuung (Levene-Test) haben, Bonferroni über beide; angenommen, wenn p > α; Ergebnis die **Schnittmenge** aller angenommenen Mengen. Der F-Test läuft über die **eigene unvollständige Betafunktion** (Kettenbruch).
- **Vergleich auf denselben Daten:** PC (stabil, Fisher-z) aus Stück 13 und NOTEARS aus Stück 14, wortgleich.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; die Läufe sind fest, ihre Zahl steht dabei. Raten sind bei 50 Läufen auf etwa ±3 bis 7 Punkte (ein Standardfehler) genau; die Richtung der Aussagen ist es, nicht die Dezimale.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (Laplace-Rauschen, n = 1 000, fünf Umgebungen mit n_e = 1 000, Seed 35) | LiNGAM richtet **6 von 6** wahren Kanten richtig (SHD 0,0); **PC richtet 4 und lässt 2 ungerichtet** (SHD 1,0); NOTEARS roh 6 von 6, standardisiert höchstens 3. ICP findet **genau {Z, X, W}**, 4 von 32 Mengen angenommen. | `test_default_sample` |
| LiNGAM nach Rauschverteilung (n = 1 000, **50 Läufe**) | **Laplace:** genau **94 %**, alle Kanten richtig gerichtet; **gleichverteilt:** genau **98 %**; **Gauß:** genau **0 %**, nur **52 %** der Kanten richtig gerichtet (Zufall), **3,6 überzählige Kanten** je Lauf, SHD 6,48. PC lässt bei jeder Verteilung 32 bis 33 % der Kanten ungerichtet. | `test_experiment_1_which_methods_leave_the_class` |
| Und standardisiert? | LiNGAM identisch (94 %, 98 %, 0 %): **skalenfrei**. NOTEARS (**12 Läufe**, Laplace): roh **100 %** richtig, standardisiert nur etwa **50 %** richtig und etwa 28 % umgekehrt (je Plattform um einzelne Kanten verschieden). | `test_experiment_1_which_methods_leave_the_class` |
| Wie nicht Gauß'sch muss das Rauschen sein? (Laplace-Mischung, n = 1 000, 50 Läufe) | Exzess-Kurtosis **0,00:** **53 %** richtig gerichtet, genau 0 %; 0,03: 55 %; **0,19** (t = 0,25): **55 %**, genau 2 %; **0,75** (t = 0,5): **92 %**, genau 70 %; **1,69:** 100 %, genau 100 %; **3,00:** 100 %, genau 90 %. Der **Übergang** liegt zwischen 0,2 und 0,75 (grobes Gitter, 50 Läufe je Zelle). | `test_experiment_2_transition_and_sample` |
| Wie viele Beobachtungen braucht LiNGAM? (50 Läufe) | Laplace: genau **48 %** (n = 100), **92 %** (300), 94 % (1 000), 96 % (3 000); gleichverteilt: 80 %, 100 %, 100 %, 98 %. | `test_experiment_2_transition_and_sample` |
| ICP: Umgebungen kumulativ (Gauß, n_e = 1 000, **50 Läufe**, α = 0,05) | Nur beobachtet und mit der Saison: **leere Menge in 100 %**; mit Faktor zugeteilt: leer 90 %; mit Wetter: **W in 100 %**, Z in 2 %, X in 8 %; mit der Nachfrage: **{Z, X, W} genau in 74 %** (Z 76 %, X 100 %, W 100 %). | `test_experiment_3_icp` |
| ICP: Stichprobe je Umgebung (fünf Umgebungen) | n_e = 100: leer **48 %**, genau 4 %; **300:** genau **18 %**, X in 50 %; **1 000:** genau **74 %**; **3 000:** genau **96 %**. α = 0,01 statt 0,05 (n_e = 1 000): genau 64 % gegen 70 % (Unterschied im Rahmen der Streuung); die Abdeckung bleibt hoch. | `test_experiment_3_icp` |
| ICP und die Rauschverteilung | Gauß 74 %, Laplace 76 % genau: **kein Unterschied** (ICP prüft Verteilungen über Umgebungen, nicht ihre Gestalt). | `test_experiment_3_icp` |
| ICP-Abdeckung (Ergebnis ⊆ wahre Eltern) | **94 bis 100 %** in allen Zeilen: Garantie von ICP bei gültigen Tests und ohne Eingriff in den Servicegrad (Irrtum höchstens α = 0,05; bei 50 Läufen streut die gemessene Rate um etwa ±3 Punkte), kein Befund. | `test_experiment_3_icp` |
| Nachfrage verborgen (LiNGAM, Laplace, n = 1 000, **30 Läufe**) | In **30 von 30** Läufen dieselbe Ausgabe: **S → X, S → Y, W → Y, X → Y, Y → C**: zwei falsche Kanten, alles gerichtet, keine Warnung. | `test_experiment_4_limits` |
| Eingriff in den Servicegrad (ICP, **30 Läufe**) | **Leere Menge in 100 %**: keine Menge ist invariant (Konstruktion der Methode); ICP nennt lieber nichts als falsche Eltern. | `test_experiment_4_limits` |
| Vorhersage von Y in einer neuen Umgebung (Saison +4, Wetter −4, Beschwerden unabhängig; Training vier Umgebungen; Rest 1,00; 30 Läufe) | Fehler: **wahre Eltern 1,00**, ohne Beschwerden 1,00, **alle Größen 1,81**, nur Mittel 3,72, **ICP-Schnittmenge allein 9,1**; von den angenommenen Mengen enthalten **52 %** die Beschwerden. Die Schnittmenge ist eine Untergrenze der Eltern, kein Vorhersagemodell. | `test_experiment_4_limits` |

Handwerte des Textes (`test_hand_values_of_the_text`): Exzess-Kurtosis Gauß 0, Laplace **+3**, gleichverteilt **−1,2**, Mischung **3 t²** (t = 0,5: 0,75); Entropien bei Varianz 1: Gauß **1,4189**, Laplace **1,3466**, gleichverteilt **1,2425**; F-Test: Gruppen (1, 2, 3) und (2, 3, 4): F = **1,5**, p = **0,2879**; I₀,₄(2, 3) = **0,5248**.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Rauschen ist nicht Gauß'sch (LiNGAM)** | Bei Gauß-Rauschen ist die Reihenfolge nicht bestimmt: eine Auswahl, mit überzähligen Kanten; schon bei kleinem nicht-gaußschem Anteil (Exzess-Kurtosis unter etwa 0,75) kippt es (Experimente 1 und 2). | Wie falsch darf das Diagramm sein? |
| **Linear, alle Größen beobachtet (LiNGAM)** | Eine verborgene Ursache gibt einen sauber gerichteten Graphen mit falschen Kanten, ohne Warnung (Experiment 4). Nichtlineare additive Rauschmodelle sind nicht gebaut. | Wann geht es gar nicht? |
| **Bekannte Umgebungen ohne Eingriff in die Zielgröße (ICP)** | Wird der Servicegrad selbst verschoben, ist keine Menge invariant: leere Menge (Experiment 4). | Wie falsch darf das Diagramm sein? |
| **Genug Stärke und Stichprobe (ICP)** | Zu wenige oder zu schwache Umgebungen, zu kleine Stichprobe: ICP nennt nur einen Teil der Eltern oder nichts (Experiment 3). | Welches Verfahren, gemessen woran? |
| **Die Schnittmenge ist keine Vorhersage (ICP)** | Sie ist eine Untergrenze der Eltern; ein Modell nur mit ihr sagt den Servicegrad schlecht voraus (Experiment 4). | nicht gebaut |
| **Ein Zielknoten, bekannte Eingriffe** | Unbekannte Eingriffe und mehrere Ziele sind nicht gebaut. | nicht gebaut |

Was **Garantie oder Konstruktion** ist, steht in `test_claims.py` als solches: die Abdeckung von ICP (Garantie, kein Befund; der Befund ist die Trefferquote), die leere Menge bei Eingriff in den Servicegrad (Konstruktion der Methode), die Kurtosis-Formel 3 t² (folgt aus den Kumulanten), und dass „Gauß: 52 % richtig gerichtet“ eine Eigenschaft des Auswahlverfahrens ist (die Reihenfolge ist bei Gauß-Rauschen nicht bestimmt), kein Befund über LiNGAM im Allgemeinen.
Der Übergang bei Exzess-Kurtosis 0,2 bis 0,75 hängt an der Entropie-Näherung (Konstanten aus dem Gedächtnis), an n = 1 000 und an der Welt; die Form (ein Übergang statt allmählichen Gleitens) ist die Aussage, nicht die Zahl. Die Umgebungen (Verschiebung +2, Faktor 2) habe ich gewählt: Die Trefferquote von ICP hängt an ihrer Stärke; in einem Nachversuch (Gauß-Rauschen, fünf Umgebungen, n_e = 1 000, 50 Läufe) gaben doppelt so starke Verschiebungen und Streuungen 94 % statt rund 70 % genau.
Bei Gauß-Rauschen entscheiden Rundungen, wohin die Auswahl fällt; bei standardisiertem NOTEARS hängt die Ausgabe an der Rechnung und weicht systematisch von der wahren Richtung ab (siehe notears-demo, nicht nur Rundung); die Zahlen können je Plattform um einzelne Läufe abweichen (die Tests gelten mit Toleranz von zwei Läufen). LiNGAM, DirectLiNGAM, die Entropie-Näherung, ICP und der Gauß-Satz (Shimizu 2006; Darmois/Skitovich) sind aus dem Gedächtnis geschrieben und nicht gegen die Quellen geprüft; die Verfahren sind durch Simulation mit bekannter Wahrheit und durch scipy (`betainc`, `f.sf`, `f_oneway`, `levene`) belegt.

## Bezug zu den Nachbarn im Portfolio

- [strukturlernen-demo](https://github.com/sebastian-hanisch/strukturlernen-demo) (Stück 13) und [notears-demo](https://github.com/sebastian-hanisch/notears-demo) (Stück 14): PC und NOTEARS stehen als Vergleichszeilen daneben (`stl_pc.py`, `stl_world.py`, `stl_constants.py`, `nts_notears.py`, `nts_baselines.py` wortgleich). PC lässt Kanten offen, NOTEARS richtet sie aus der Skala, LiNGAM aus der Gestalt des Rauschens, ICP aus Umgebungen.
- [ica-demo](https://github.com/sebastian-hanisch/ica-demo): FastICA trennt gemischte Quellen durch Nicht-Gaußianität; LiNGAM nutzt dieselbe Idee für Richtungen. Hier DirectLiNGAM (kein Entmischen, kein ICA), die Nicht-Gaußianität ist das einzige Band.
- **Experimente und Wirkungsmessung:** Eingriffe sind dort Experimente mit Zuteilung; hier sind sie Umgebungen, die nur zum Entdecken der Eltern dienen (ohne Wirkungsschätzung).

## Tests

Hand-Instanzen (Exzess-Kurtosis 3 t², Entropie 1,4189, F-Test 1,5 und 0,2879, Beta 0,5248, ICP-Mini-Beispiel Y = X + ε), die unvollständige Betafunktion und der F-Test gegen scipy, der Invarianztest gegen `f_oneway` und `levene` (Median), die Schnittmenge gegen eine zweite Rechnung über scipy-p-Werte, DirectLiNGAM auf Zufalls-DAGs mit bekannter Wahrheit und vertauschten Spalten, Gauß-Rauschen, Skalenfreiheit, Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Umgebungen als Schalter, Experimente, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py` | d-Separation und Äquivalenzklassen (wortgleich aus Stück 2) |
| `stl_pc.py`, `stl_world.py`, `stl_constants.py` | PC, Welt und Konstanten aus Stück 13 (wortgleich) |
| `nts_notears.py`, `nts_baselines.py` | NOTEARS und Vergleichsmaße aus Stück 14 (wortgleich) |
| `lci_world.py` | sechs Größen, Rauschfamilien, Umgebungen |
| `lci_lingam.py` | DirectLiNGAM: Entropie-Näherung, exogene Größe, Reihenfolge, Kanten, Paar |
| `lci_icp.py` | unvollständige Betafunktion, F- und Levene-Test, Invarianztest, Schnittmenge |
| `lci_evaluation.py` | Einstellungen, zwei Analysen, vier Experimente, Verdikte |
| `lci_visualization.py`, `lci_pdf_export.py` | Figuren, Tabellen, PDF |
| `lci_constants.py`, `lci_presets.py` | Regler, Umgebungen, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **ICA-LiNGAM** (FastICA und Permutationssuche): dieselbe Nicht-Gaußianität wie `ica-demo`, hier DirectLiNGAM.
- **Nichtlineare additive Rauschmodelle, latente Varianten von LiNGAM, unbekannte Eingriffe, mehrere Zielgrößen:** nur im Grenzen-Absatz genannt.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
