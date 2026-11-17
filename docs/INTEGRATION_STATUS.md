# Integrationsstand vom 17.11.2026

## Repository und Ausgangsbasis

Gemeinsames Repository: [vika17oxy/MVS](https://github.com/vika17oxy/MVS).
Die Integration baut direkt auf Viktoriias Commit
[`ec04d2e` – Add GTSRB classification and evaluation](https://github.com/vika17oxy/MVS/commit/ec04d2ea828f7d924307acae2faaf81fb26c624b).
Viktoriias Klassifikationscode bleibt unverändert.

Die folgenden Festlegungen wurden aus dem vorhandenen Code, den Tests und
Manifesten rekonstruiert. Sie sind die überprüfbare Grundlage dieser Fortsetzung.

## Bereits erledigt – nicht erneut aufbauen

- Strikter OpenCV-Detektor integriert; `localizer.py` und `preprocess.py`
  sind bytegleich zum abgeschlossenen Starter-Stand. NMS bleibt bei 0,25.
- Adapter `src/detection.py`: exklusive xyxy-Grenzen zu xywh ohne `+1`.
- `src/pipeline.py`: mehrere vorhergesagte BGR-Crops, Kopien zum Schutz des
  Originalbilds, Klassenvalidierung, Ablehnung `-1`, separater Klassifikationsscore.
- Viktoriias Klassifikatoren, Features und GTSRB-Konfiguration unverändert.
  Keine Vermischung ihrer 12-Klassen-GTSRB-Auswertung mit MVSR-Endmetriken.
- TRAIN-Loader, Klassenmapping und reproduzierbarer Split vorhanden:
  541 Trainingsbilder mit 698 Zeichen, 136 Entwicklungsbilder mit 172 Zeichen.
  Gruppen nach vollständigen decodierten Bildpixeln; Seed 42.
- Template-Modell mit allen 43 Klassen vorhanden:
  `models/mvsr_template/{model.joblib,manifest.json}`.
- SVM-Modell mit 38 Klassen vorhanden:
  `models/mvsr_svm_partial/{model.joblib,manifest.json,cv_results.json}`.
- Detektorbenchmark auf 677 TRAIN-Bildern vorhanden: TP 556, FP 528, FN 314;
  Precision 51,29 %, Recall 63,91 %. Nicht erneut ausgeführt.
- Template-Entwicklungsauswertung und Sample-Demo bereits vorhanden.
  Modelle wurden in dieser Fortsetzung nicht neu trainiert.

## In dieser Fortsetzung abgeschlossen

1. Ausgangscommit und Integrationsumfang geprüft.
2. Modellhash des SVM sowie gleiche Datenhashes und Train-/Dev-Partitionen
   beider Modelle geprüft. Zum Auswertungszeitpunkt entsprach der Quellcode
   den bisherigen Manifesten. Die späteren CLI-Pfad- und Hinweisänderungen
   verändern keine Detektor-, Klassifikations- oder Metrikberechnung.
3. Die noch fehlende Entwicklungsauswertung des gespeicherten SVM ausgeführt:

   ```powershell
   .venv\Scripts\python.exe -B evaluate_mvsr.py --dataset "PFAD_ZU_DATASET" --model-dir models/mvsr_svm_partial --output artifacts/mvsr_svm_partial_dev --allow-partial-classes
   ```

4. Vergleich geprüft: beide Auswertungen verwenden dieselben 136 Bilder,
   dieselben geordneten Detektorboxen, dieselben Quellcodehashes und Paketversionen.
   Summen der klassenweisen TP/FP/FN stimmen mit den Gesamtmetriken überein;
   beide Confusion Matrizen enthalten 270 Zuordnungen einschließlich Hintergrund
   und übersehener Zeichen.
5. Vorhandene Tests zur aktuellen Integration ausgeführt: **51 bestanden**.
   Geprüft werden unter anderem echte Detektion mit Template-Klassifikation,
   BGR-Crops, Bildgrenzen, Ablehnung, Klassenabdeckung, Split und Systemmetriken.
6. README und diese Übergabe ergänzt. Eine kleine maschinenlesbare
   Ergebniszusammenfassung liegt in [integration_results.json](integration_results.json).

Die Funktionsprüfung ergab keinen notwendigen Eingriff in den bestehenden
Detektor oder Viktoriias Klassifikator. Die neue Arbeit schließt die offene
SVM-Validierung und die bisher fehlende gemeinsame Dokumentation ab.

Für die Integration wurden die CLI-Standardpfade vereinheitlicht:
beide Befehle verwenden `artifacts/inference/` relativ zum Repository und
verlangen `--dataset`. Die Lokalisierungsauswertung verweist für Klassenmetriken
und Confusion Matrix nun auf die vorhandene `evaluate_mvsr.py`.
Danach bestanden erneut alle **51 Unit-Tests**. Zusätzliche CLI-Integrationstests
in einem temporären Projekt bestätigten die Standardpfade beim Aufruf aus einem
anderen Arbeitsverzeichnis, explizite relative Pfade, den verpflichtenden
Datensatzparameter, Inferenz mit beiden gespeicherten Modellen einschließlich
der SVM-Teilklassenfreigabe sowie die ausdrückliche Teilmengenauswertung.
Die temporären Bilder und Ergebnisse wurden entfernt; bestehende Laufartefakte
wurden nicht überschrieben.

## Vergleich auf denselben Entwicklungsbildern

IoU ≥ 0,5; klassenblindes greedy 1:1-Matching vor Ablehnung.
Template-Schwelle 0,3 NCC; SVM-Schwelle 0,5 Wahrscheinlichkeit.
Die Schwellen wurden nicht anhand dieses Vergleichs verändert.

| Messgröße | Template, 43 trainierte Klassen | SVM, 38 trainierte Klassen |
|---|---:|---:|
| Detektor TP / FP / FN | 110 / 98 / 62 | 110 / 98 / 62 |
| Korrekte Klassen auf 110 zugeordneten Kandidaten | 76 (69,09 %) | 83 (75,45 %) |
| Korrekte Klassen auf 172 GT-Crops, mit Ablehnung | 120 (69,77 %) | 129 (75,00 %) |
| GT-Crop-Accuracy ohne Ablehnung | 69,77 % | 90,70 % |
| Akzeptierte / abgelehnte Kandidaten | 175 / 33 | 142 / 66 |
| End-to-End TP / FP / FN | 76 / 99 / 96 | 83 / 59 / 89 |
| End-to-End Precision | 43,43 % | 58,45 % |
| End-to-End Recall | 44,19 % | 48,26 % |
| End-to-End F1 | 43,80 % | 52,87 % |

GT-Crop-Ergebnisse verwenden perfekte Ground-Truth-Boxen und messen nicht
die komplette Straßenbild-Pipeline. Auch die 90,70 % ohne Ablehnung sind
keine End-to-End-Accuracy.

Lokale Detaildateien liegen jeweils unter `artifacts/mvsr_template_dev/`
und `artifacts/mvsr_svm_partial_dev/`: `metrics.json`, `predictions.json`,
`per_class.csv`, `confusion_matrix.csv`, `split_manifest.json`.
Diese umfangreichen Laufartefakte und Modelle bleiben durch `.gitignore` lokal.

## Grenzen und nächste offene Arbeit

Das SVM fehlt für **19, 24, 27, 31, 37**. Diese Klassen haben im
Trainingsanteil nur 1, 2, 1, 1 bzw. 1 Crop und zu wenige unabhängige Gruppen
für Viktoriias bestehende 3-fache Kreuzvalidierung. Das SVM bleibt deshalb
ein ausdrücklich gekennzeichnetes Teilklassen-Experiment. Das Template-Modell
bleibt der verfügbare Integrationsweg mit allen 43 Klassen.

Die Entwicklungsmenge enthält Ground Truth für nur **30 Klassen**. Ohne
Ground Truth sind 0, 16, 19, 20, 21, 24, 27, 29, 31, 32, 36, 37 und 41.
Der Split reserviert bis zu drei Bildgruppen je Klasse im Training; seltene
Klassen bleiben dadurch vollständig dort. Bei 16 und 36 führt die zufällige
Auswahl ganzer Bildgruppen unter diesen Reservebedingungen ebenfalls zu
keinen Entwicklungsbeispielen. Abdeckung aller Klassen im Dev-Anteil ist nicht garantiert.
Insbesondere werden die fünf fehlenden SVM-Klassen hier nicht geprüft;
aus dem besseren SVM-Wert folgt keine vollständige 43-Klassen-Abdeckung.

Der Detektor wurde auf ganz TRAIN entwickelt. Die gemeinsame Auswertung ist
deshalb ein Entwicklungsvergleich, kein unabhängiger Test der Gesamtpipeline.
Pixelidentische Bilder werden gruppiert; unterschiedliche Aufnahmen desselben
physischen Zeichens können mangels Track-IDs nicht sicher zusammengehalten werden.
Held-out wurde in dieser Fortsetzung nicht geöffnet oder ausgewertet.

Noch offen für die gemeinsame Abgabe:

- Für ein vollständiges 43-Klassen-SVM müssen die seltenen Klassen mit
  ausreichend unabhängigen Trainingsgruppen versorgt oder das Trainingsverfahren
  gemeinsam neu festgelegt werden. Den aktuellen Klassenfilter nicht stillschweigend entfernen.
- Vor einer abschließenden Held-out-Auswertung Modell und Schwellen festlegen;
  die jetzigen Entwicklungsbefehle sind bewusst auf TRAIN beschränkt.
- Gemeinsame Präsentation, Demo-Video und tatsächlicher Ubuntu-Lauf.
