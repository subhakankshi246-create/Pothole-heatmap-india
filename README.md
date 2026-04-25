# Pothole Heatmap — India Road Health Monitor

AI-powered crowdsourced road monitoring system that detects potholes from 
dashcam footage and generates a real-time priority heatmap with automated 
PWD alerts for municipal authorities across Indian cities.

## Architecture
![Architecture](architecture.svg)

Dashcam Video Frames
↓
YOLOv8n (trained on BharatPotHole dataset - Kaggle GPU)
↓
Pothole Detections + GPS coordinates
↓
Delta Lake (raw ingestion via Apache Spark)
↓
Severity Scoring Engine (Spark SQL)

Road condition indicators (texture, cracks, ruts)
MoRTH accident history weighting
Road type priority multiplier
↓
Priority Labels: CRITICAL / HIGH / MONITOR
↓
MLflow Model Registry + Databricks SQL Dashboard
↓
Auto-generated PWD Notice (Motor Vehicles Act 1988)

## Databricks Components Used
- **Delta Lake** — stores raw road sections, scored results, YOLO detections, PWD alerts
- **Apache Spark SQL** — severity scoring engine joining road conditions and accident data
- **MLflow** — model registry for YOLOv8n pothole detector
- **Databricks SQL Dashboard** — live priority heatmap
- **Serverless Compute** — all notebooks run serverless

## Datasets
- BharatPotHole (Kaggle) — 7,000+ Indian dashcam frames in YOLO format
- Synthetic road sensor data — 50,000 sections across 6 cities generated using PySpark
- Severity scoring inspired by Abed et al. (2023) pavement condition research

## How to Run (Step by Step)

### Prerequisites
- Databricks account (free tier works)
- Kaggle account (free)
- Python 3.10+

### Step 1 — Databricks Setup
1. Go to community.cloud.databricks.com
2. Open SQL Editor → run:
   CREATE DATABASE IF NOT EXISTS pothole_heatmap;
3. Create new notebook → attach to Serverless compute
4. Copy paste 00_setup_python.py → Run All cells
   Expected output: "✅ Delta Lake is working — data is queryable!"
   Time: ~3 minutes

### Step 2 — Train YOLO Model (Kaggle)
1. Go to kaggle.com → New Notebook
2. Add dataset: surbhisaswatimohanty/bharatpothole
3. Enable GPU T4 accelerator (Settings → Accelerator)
4. Paste code from notebooks/kaggle_training.py
5. Run all cells
   Expected output: mAP50 ~0.39, pothole_best.pt in output
   Time: ~25 minutes

### Step 3 — Upload Model to Databricks
1. Download best.pt from Kaggle output tab
2. Databricks → Data Ingestion → Upload
3. Create volume: workspace/default/pothole_models
4. Upload best.pt → path: /Volumes/workspace/default/pothole_models/best.pt

### Step 4 — Run MLflow Notebook
1. Open 01_mlflow_inference.py in Databricks
2. Run all cells (skip Cell 1 pip install — not needed on Serverless)
   Expected output: "✅ Model logged to MLflow!"

### Step 5 — Run Severity Scoring
1. Open 02_severity_scoring.py in Databricks
2. Run all cells
   Expected output: PWD alerts table saved, sample notice printed

### Step 6 — Run Random Forest
1. Open 03_random_forest_ml.py in Databricks
2. Run all cells
   Expected output: AUC-ROC ~0.639, feature importance printed

### Step 7 — View Dashboard
## Live Dashboard
[Pothole Heatmap - India Road Health Monitor]((https://dbc-8785c366-12b2.cloud.databricks.com/dashboardsv3/01f14026fa1f1ac7ad349d522e4428cf/published?o=7474647709257499))

## Demo Steps for Judges
1. Open dashboard link above → observe 47K critical sections
2. Open 02_severity_scoring notebook → scroll to Cell 5 → 
   read auto-generated PWD notice
3. Go to Experiments (left sidebar) → pothole_heatmap_experiment → 
   click the run → verify mAP50=0.392 logged
4. Open 03_random_forest_ml → Cell 5 → verify feature importance table


## Model Performance
| Metric | Value |
|--------|-------|
| mAP50 | 0.392 |
| Precision | 0.497 |
| Recall | 0.421 |
| Training data | 7,000+ Indian dashcam frames |
| Training platform | Kaggle Tesla T4 GPU |

## Tech Stack
- Databricks Serverless, Delta Lake, Spark SQL, MLflow
- YOLOv8n (Ultralytics)
- Python, PySpark
- Kaggle Tesla T4 GPU for training
