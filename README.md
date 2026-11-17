# MVS – gemeinsame MVSR-Pipeline

Alis klassischer OpenCV-Detektor ist mit Viktoriias Klassifikatoren verbunden:
Straßenbild (BGR) → automatisch lokalisierte Boxen → unveränderte BGR-Crops →
Klasse oder Ablehnung. Die Verarbeitung verwendet keine neuronalen Netze.

Der lokale Integrationsstand und die gemessenen Ergebnisse stehen in
[docs/INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md).

## Bestehenden Stand weiterverwenden

Die Befehle werden im Repository `MVS` ausgeführt. In der bestehenden lokalen
Arbeitskopie sind `.venv`, die Modelle und die bisherigen Ergebnisse bereits
vorhanden. Training und abgeschlossene Benchmarks müssen nicht erneut laufen.
Modelle, Datensätze und neue Laufartefakte sind nicht Teil des Git-Repositories.

Für einen neuen Demo-Lauf mit dem vorhandenen 43-Klassen-Modell:

```powershell
.venv\Scripts\python.exe run.py --dataset "PFAD_ZU_DATASET" --model models/mvsr_template/model.joblib --output artifacts/demo_template_neu --limit 10
```

`PFAD_ZU_DATASET` bezeichnet einen Ordner mit `train/images/`. Inferenz liest
keine Annotationen. Die JSON-Datei enthält alle Kandidaten einschließlich
Ablehnungen; annotierte Bilder zeigen die Klasse und den Klassifikationsscore.
Demo-Bilder können zum Training gehört haben; sie sind kein Qualitätsnachweis.
Für jeden neuen Lauf einen neuen Ausgabeordner verwenden.

Nur Lokalisierung, mit dem bestehenden strikten Detektor:

```powershell
.venv\Scripts\python.exe run.py --dataset "PFAD_ZU_DATASET" --output artifacts/demo_detection_neu --mode strict --limit 10
```

## Schnittstellen

`run.py` und `evaluate_detection.py` verlangen beide `--dataset`.
Ohne weitere Pfadangaben schreibt `run.py` nach `artifacts/inference/`
im Repository; `evaluate_detection.py` liest dort `detections.json` und
schreibt daneben `detection_metrics.json`. Diese Standardpfade bleiben auch
beim Aufruf aus einem anderen Arbeitsverzeichnis gleich.
Explizite relative Pfade beziehen sich weiterhin auf das Arbeitsverzeichnis.

```powershell
.venv\Scripts\python.exe run.py --dataset "PFAD_ZU_DATASET"
.venv\Scripts\python.exe evaluate_detection.py --dataset "PFAD_ZU_DATASET"
```

Bei einem begrenzten Lauf (`--limit`) benötigt die Lokalisierungsauswertung
zusätzlich `--allow-partial`. Klassenmetriken und Confusion Matrix werden
separat mit `evaluate_mvsr.py` berechnet.

- `src.detection.detect_signs(image_bgr)` liefert `(x, y, width, height)`.
- `src.pipeline.detect_and_classify(image_bgr, classifier)` liefert pro Box
  `x1`, `y1`, `x2`, `y2`, `class_id`, `classification_confidence`.
- `x2/y2` sind exklusiv. Crops werden kopiert; der Klassifikator erhält BGR.
- Viktoriias Vertrag bleibt `predict(roi_bgr) -> (class_id, confidence)`.
  `class_id=-1` bedeutet Ablehnung, gültige Klassen bleiben unverändert 0–42.
- Lokalisierung allein verwendet weiterhin die unveränderliche `Detection`
  mit `class_id=None`. Ihr `confidence` ist eine Detektorheuristik.
  `classification_confidence` ist separat: Template-NCC bzw. ML-Wahrscheinlichkeit.
- Vor einem Lauf prüft `validate_classifier()` die vollständige Klassenabdeckung.
  Teilklassenmodelle benötigen ausdrücklich `--allow-partial-classes`.

Die ursprünglichen GTSRB-Skripte `train.py` und `evaluate.py` sowie
`src/config.py` mit Viktoriias 12-Klassen-Auswahl bleiben eigenständig.
Die MVSR-Integration verwendet `train_mvsr.py` und `evaluate_mvsr.py`.

## Auswertung und Datenaufteilung

Das MVSR-Format enthält `classes.csv`, `train/images/` und
`train/labels.csv` mit `filename,x1,y1,x2,y2,class_id`.
`src/mvsr_classes.csv` hält die geprüfte Klassenbelegung fest.

Das vorhandene Training teilt 677 TRAIN-Bilder in 541 Trainings- und
136 Entwicklungsbilder. Crops desselben Bildes und pixelidentische Bilder
bleiben in derselben Partition. Seltene Klassen bleiben im Training.
Das Manifest speichert Dateihashes und die genaue Aufteilung.

Abgeschlossene Auswertungen sind lokal verfügbar:

- `artifacts/mvsr_template_dev/`: vollständiges Template-Modell.
- `artifacts/mvsr_svm_partial_dev/`: diagnostisches SVM mit 38 Klassen.
- Jeweils `metrics.json`, `predictions.json`, `per_class.csv`,
  `confusion_matrix.csv`, `split_manifest.json`.

Für eine spätere, bewusst neue Auswertung eines geänderten Modells:

```powershell
.venv\Scripts\python.exe evaluate_mvsr.py --dataset "PFAD_ZU_DATASET" --model-dir models/mvsr_template --output artifacts/template_dev_neu
```

`evaluate_mvsr.py` bewertet ausschließlich die zum Modell gespeicherten
Entwicklungsbilder und prüft Daten- und Modellhashes. Es trennt Lokalisierung,
Klassifikation auf zugeordneten vorhergesagten Crops, Klassifikation auf
Ground-Truth-Crops und End-to-End-Metriken. Matching: klassenblind, greedy,
eins zu eins, IoU ≥ 0,5; Ablehnungen bleiben beim Matching enthalten.
Falsche Klassen zählen als FP der vorhergesagten und FN der tatsächlichen Klasse.
Die Confusion Matrix enthält zusätzlich Hintergrund, Ablehnung und übersehene Zeichen.

Der Detektor wurde zuvor auf ganz TRAIN entwickelt. Diese Ergebnisse sind
Entwicklungswerte, kein unabhängiger Gesamtsystemtest. Physische Zeichen über
unterschiedliche Aufnahmen lassen sich ohne Track-IDs nicht sicher gruppieren.
`heldout` bleibt für die spätere abschließende Bewertung reserviert.

## Tests und neue Umgebungen

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Nur bei einer neuen Arbeitskopie: Python 3.12 verwenden, eine virtuelle
Umgebung erstellen und `pip install -r requirements.txt` darin ausführen.
Unter Linux heißt der Interpreter `.venv/bin/python`; ein Ubuntu-Lauf ist
bisher nicht belegt. Lokale Modelle werden bei einer neuen Arbeitskopie nicht
mitgeliefert; `train_mvsr.py --help` beschreibt deren TRAIN-basierten Aufbau.
