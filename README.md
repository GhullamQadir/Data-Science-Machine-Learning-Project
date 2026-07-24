# Talent Scout Engine

**Advanced Data Science and Machine Learning Framework for Next-Generation Cricket Scouting**

Live Application Deployment: [https://data-science-machine-learning-project.streamlit.app/](https://data-science-machine-learning-project.streamlit.app/)

---

## Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Motivation and Problem Statement](#motivation-and-problem-statement)
- [Architecture](#architecture)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Model](#machine-learning-model)
- [Dashboard and Interface](#dashboard-and-interface)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Model Retraining](#model-retraining)
- [Configuration](#configuration)
- [Academic Context](#academic-context)
- [Authors](#authors)

---

## Live Demo

The Talent Scout Engine interactive application is deployed and live on Streamlit Cloud:

**Live Dashboard URL:** [https://data-science-machine-learning-project.streamlit.app/](https://data-science-machine-learning-project.streamlit.app/)

Users can access the live application to filter players, visualize performance clusters, and run real-time ML scouting inference without any local setup.

---

## Project Overview

Talent Scout Engine is a Data driven scouting platform built as a 4th Semester academic project in Data Science and Machine Learning. The system ingests raw cricket performance data from ICC international tournaments and T20 franchise leagues (2024–2026 cycles), applies a structured feature engineering pipeline, and trains an unsupervised K-Means clustering model to segment players into objectively ranked performance tiers.

The output is an interactive Streamlit web dashboard that allows scouts, analysts, and coaches to filter, compare, and evaluate player profiles across multiple dimensions batting, bowling, fielding, and recent form with a particular emphasis on identifying emerging Under-19 and Under-23 talent.

---

## Motivation and Problem Statement

Traditional cricket scouting relies heavily on manual observation and subjective judgment. This approach is:

- Slow and costly to scale across multiple tournaments simultaneously
- Prone to selection bias and regional blind spots
- Unable to synthesize multi-dimensional performance indicators into a unified ranking

This project addresses these limitations by applying unsupervised machine learning to derive objective, data-backed performance tiers. The composite Scout Rating quantifies a player's overall value across batting, bowling, fielding, and formm eliminating reliance on gut-feel assessments.

---

## Architecture

The system follows a four-stage linear pipeline:

```
Raw Excel / CSV Data
        |
        v
  Data Ingestion Layer
  (smart_load_file / load_and_merge_data)
        |
        v
  Feature Engineering Pipeline
  (feature_engineering in src/data_pipeline.py)
        |
        v
  K-Means Clustering Model
  (train_and_persist_model)
        |
        v
  Streamlit Interactive Dashboard
  (app.py)
```

**Ingestion Layer:** Handles multi sheet Excel workbooks with stacked batting, bowling, and fielding sections. Auto detects header rows, De duplicates columns, resolves naming variations across tournaments, and merges data by player name.

**Feature Engineering:** Constructs domain-specific impact scores from raw statistics.

**Clustering Model:** Groups players into three ranked performance tiers (Tier 1 = Elite, Tier 2 = Developing, Tier 3 = Emerging).

**Dashboard:** Renders the processed output as a filterable, sortable interface with real-time predictor capability.

---

## Feature Engineering

All impact scores are computed from raw tournament statistics using the following formulas:

**Batting Impact**
```
Batting_Impact = (Total_Runs * 0.40) + (Batting_Avg * 0.30) + (Strike_Rate * 0.30)
```

**Bowling Impact**
```
Bowling_Impact = (Total_Wickets * 0.50 + (100 / (Bowling_Avg + eps)) * 0.25 + Economy_Rate) * 0.25
```

**Fielding Impact**
```
Fielding_Impact = (Total_Catches * 0.40) + (Total_Stumpings * 0.30) + (Total_RunOuts * 0.30)
```

**Composite Scout Rating**
```
Scout_Rating = (Batting_Impact * 0.35) + (Bowling_Impact * 0.35) + (Fielding_Impact * 0.10) + (Form_Index_Last5 * 2.0)
```

All three impact scores are independently scaled to a 0–100 range using MinMaxScaler before the composite rating is computed. The Scout Rating is then re-scaled to 0–100 for uniform comparison.

**Form Index:** A proxy for recent form derived from the player's batting average relative to the dataset maximum, mapped to a 2–9.5 scale.

**Role Classification:** Assigned algorithmically based on the ratio of batting score to bowling score, with an All-Rounder designation applied when neither discipline dominates by more than 20%.

---

## Machine Learning Model

**Algorithm:** K-Means Clustering (scikit-learn)

**Rationale for unsupervised learning:** Player performance data from these tournaments carries no pre-existing ground-truth labels. K-Means allows the algorithm to discover natural groupings without injecting human bias through predefined labels.

| Parameter | Value |
|-----------|-------|
| n\_clusters | 3 |
| random\_state | 42 |
| n\_init | 10 |
| Feature set | Batting\_Impact, Bowling\_Impact, Fielding\_Impact, Form\_Index\_Last5, Scout\_Rating |

**Tier Assignment:** After clustering, each cluster is assigned a tier label based on the mean Scout Rating of its members. The cluster with the highest mean rating is designated Tier 1 (Elite), followed by Tier 2 (Developing) and Tier 3 (Emerging). This ensures tier labels are always interpretable regardless of how K-Means internally numbers the clusters.

**Model Persistence:** The trained KMeans object is serialized using `joblib` and saved as `models/talent_scout_model.pkl`. Processed output is cached as `data/processed/processed_data.csv`.

---

## Dashboard and Interface

The Streamlit dashboard (`app.py`) provides the following capabilities:

**Dynamic Data Loading**
- Upload any `.csv` or `.xlsx` dataset directly from the sidebar
- Automatic schema normalization for datasets with varied column naming conventions
- Falls back to cached processed data or re-runs the full pipeline against the default raw dataset

**Sidebar Filters**
- Country / national pool selection
- Player role filter (Batsman, Bowler, All-Rounder)
- Performance Tier filter (Tier 1, Tier 2, Tier 3)
- Year filter (2024, 2025, 2026, or All)
- Under-19 and Under-23 isolation toggle

**Summary Metrics**
- Total scanned player assets
- Count of Tier 1 (Elite) prospects
- Mean Scout Rating across the filtered pool
- Global Squad Form Index

**Visualizations**
- 2D Performance Impact Scatter plot (Batting Impact vs. Bowling Impact, sized by Scout Rating, colored by Tier)
- Interactive Plotly charts with full hover metadata

**Real-Time Predictor**
- Input a new player's raw statistics via form controls
- Instantly computes Scout Rating and predicts the K-Means cluster assignment
- Falls back to a rule-based rating engine if the model file is not present on disk

**Theme**
- Dual-mode UI: Light and Dark themes with a toggle, powered by custom CSS injected at runtime using Google Fonts (Outfit family)

---

## Project Structure

```
Talent_Scout_Engine/
|
|-- app.py                        # Main Streamlit application
|-- requirements.txt              # Python package dependencies
|-- STARTUP_GUIDE.md              # Quick-start command reference
|-- .gitignore
|
|-- src/
|   |-- __init__.py
|   `-- data_pipeline.py          # Feature engineering and model training functions
|
|-- models/
|   |-- talent_scout_model.pkl    # Serialized K-Means model (generated at runtime)
|   `-- talent_scout_model.py     # Model utility reference
|
|-- data/
|   |-- Raw_data/
|   |   `-- Raw Dataset For DS & ML Project.xlsx   # Source tournament data
|   |-- raw/                      # Alternative raw data directory
|   `-- processed/
|       `-- processed_data.csv    # Pipeline output cache (generated at runtime)
|
|-- retrain_model.py              # Standalone model retraining script
|-- generate_sample_data.py       # Synthetic data generation utility
|-- inject_ages.py                # Age enrichment utility
|-- fix_excel.py                  # Excel preprocessing utility
|-- fix_u19.py                    # Under-19 data correction utility
|-- inspect_cols.py               # Column inspection diagnostic script
|
`-- .streamlit/                   # Streamlit server configuration
```

---

## Dataset

**Source File:** `data/Raw_data/Raw Dataset For DS & ML Project.xlsx`

**Coverage:**
- ICC international tournaments (2024–2026)
- T20 franchise leagues
- Under-19 World Cup and bilateral series
- Under-23 emerging tournaments

**Data Dimensions:**
- Multi-sheet workbook with separate sheets per tournament per discipline (batting, bowling, fielding)
- Player-level aggregated statistics including: runs, batting average, strike rate, wickets, bowling average, economy rate, catches, stumpings, run-outs
- Supplementary age data sourced from `Data Set/Age Of Players.txt` for Under-19 and Under-23 identification

**Preprocessing:**
- Auto-detection of header rows embedded within sheets
- Column name normalization and deduplication
- Outer join merge across batting, bowling, and fielding dimensions by player name
- Missing value imputation with domain-appropriate defaults (e.g., Bowling\_Avg defaults to 30, Economy\_Rate defaults to 8)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.x |
| Web Framework | Streamlit |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn (KMeans, MinMaxScaler) |
| Model Serialization | joblib |
| Visualization | Plotly Express |
| Excel I/O | openpyxl |
| Statistical Analysis | scipy, statsmodels |
| Plotting Utilities | matplotlib, seaborn |
| UI Typography | Google Fonts — Outfit |

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip

### First-Time Setup

**1. Navigate to the project directory**

```bash
cd "Project_Main_File/Talent_Scout_Engine"
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the virtual environment**

Windows:
```bash
venv\Scripts\activate
```

macOS / Linux:
```bash
source venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the data pipeline** (generates processed data and trains the model)

```bash
python src/data_pipeline.py
```

**6. Launch the dashboard**

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your default browser.

---

## Usage

### Standard Workflow

1. Launch the dashboard with `streamlit run app.py`
2. The application auto-loads the pre-processed dataset from `data/processed/processed_data.csv`
3. If no cached data is present, the pipeline runs automatically against the default raw dataset
4. Use the sidebar to filter players by country, role, tier, age group, and year
5. Analyze the scatter plot to identify high-impact players in the 2D performance space

### Custom Dataset Upload

1. Click the Upload Player Dataset control in the sidebar
2. Select a `.csv` or `.xlsx` file
3. The system normalizes column names, runs feature engineering, trains a new K-Means model on your data, and refreshes the dashboard automatically

**Minimum required column:** `Player_Name` (or any recognized alias: `Player`, `Name`, `Cricketer`)

**Optional recognized columns:** `Country`, `Age`, `Runs`, `Batting_Avg`, `Strike_Rate`, `Wickets`, `Bowling_Avg`, `Economy_Rate`, `Catches`, `Stumpings`, `Run_Outs`

### Real-Time Predictor

1. Scroll to the Real-Time Machine Learning Scouting Predictor section
2. Enter a player's statistics using the numeric inputs and form slider
3. Click Run Analytics Inference Engine
4. The system computes a Scout Rating and assigns the player to a K-Means cluster

---

## Model Retraining

To retrain the model against the raw Excel dataset from scratch (useful after updating the source data file):

```bash
python retrain_model.py
```

This script:
1. Reads all `.xlsx` files from `data/Raw_data/`
2. Detects batting, bowling, and fielding sheets by name convention (`_BAT`, `_BOWL`, `_FIELD` suffixes)
3. Applies the full feature engineering pipeline
4. Trains a fresh K-Means model with 3 clusters
5. Saves the model to `models/talent_scout_model.pkl`
6. Saves the processed output to `data/processed/processed_data.csv`
7. Prints the top 5 fielders by total dismissals as a verification output

---

## Configuration

**Model parameters** are defined inline in `src/data_pipeline.py` within the `train_and_persist_model` function. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| n\_clusters | 3 | Number of performance tiers |
| random\_state | 42 | Reproducibility seed |
| n\_init | 10 | Number of K-Means initialization runs |
| Scout Rating weights | Bat 35%, Bowl 35%, Field 10%, Form 20% | Composite rating formula |

**File paths** are resolved dynamically relative to the script location using `pathlib.Path`, making the project portable across machines without configuration changes.

---

## Academic Context

**Degree Program:** Bachelor of Science in Artificial Intelligence (BSAI)  
**Semester:** 4th Semester  
**Courses:** Introduction to Data Science / Machine Learning  
**Project Type:** Sports Related Project  
**Academic Year:** 2026

This project demonstrates applied knowledge of:
- Data wrangling and preprocessing with pandas
- Domain-driven feature engineering for sports analytics
- Unsupervised machine learning with K-Means clustering
- Feature scaling and normalization
- Model persistence and deployment
- Interactive data visualization and dashboard development with Streamlit and Plotly

---

## Authors

Developed as a collaborative academic project for the 4th Semester Data Science and Machine Learning course.

---

*This project is developed for academic and educational purposes.*
