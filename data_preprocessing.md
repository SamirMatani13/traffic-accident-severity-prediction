# Traffic Accident Severity Prediction — Data Preprocessing

Predicting accident **severity (1–4)** from the US Accidents dataset. This repo gives you a **config-driven preprocessing CLI** that downloads data from Kaggle, runs feature engineering, and saves **train/val/test** artifacts for modeling.

---

## 1) Requirements

- **Python** 3.11+ (3.12 recommended)
- **Windows PowerShell** (or any shell)
- **Kaggle API** credentials  
  - Get `kaggle.json`: Kaggle → Account → *Create New API Token*  
  - Save to: `%HOMEPATH%\.kaggle\kaggle.json`  
  - (If needed) `chmod` equivalent is not required on Windows.

---

## 2) Setup

You created `.venv` yourself — perfect. Keep your Makefile like this:

```make
setup:
	pip install -U pip && .\.venv\Scripts\python -m pip install -r requirements.txt
````

Run once from the **repo root**:

```powershell
make setup
```

> If `make` isn’t installed, you can always run:
> `.\.venv\Scripts\python -m pip install -r requirements.txt`

---

## 3) Project Structure

```
traffic-accident-severity-prediction/
├─ Makefile
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ raw/         # Kaggle CSV lands here
│  ├─ interim/
│  └─ processed/   # outputs (npz/csv/joblib/json)
├─ reports/
│  └─ figures/
├─ scripts/
│  ├─ fetch_kaggle.py
│  └─ preprocess.py
└─ src/
   └─ traffic_severity/
      ├─ __init__.py
      ├─ config.py
      ├─ preprocess.py
      └─ utils.py
```

---

## 4) Configuration (quick knobs) — `configs/default.yaml`

```yaml
# Paths
raw_csv_path: "data/raw/US_Accidents_March23.csv"  # set to the actual CSV
output_dir: "data/processed"
reports_dir: "reports"

# Columns
target_column: "Severity"
id_columns: ["ID"]
datetime_columns: ["Start_Time", "End_Time"]
allowlist_only: true   # keep only selected features (good for RAM)
selected_features:
  - Severity
  - Start_Time
  - End_Time
  - Distance(mi)
  - Start_Lat
  - Start_Lng
  - Weather_Condition
  - Temperature(F)
  - Visibility(mi)
  - Wind_Speed(mph)
  - Precipitation(in)
  - Sunrise_Sunset
  - Traffic_Signal
  - Junction
  - Crossing

# Feature engineering
make_duration: true        # End - Start -> duration_min
time_features: {hour: true, dow: true, month: true}
cyclical_encode: true      # sin/cos encodings

# Sampling (start small; bump later)
sample_n_rows: 100000      # 100k recommended on CPU to start
sample_fraction: null

# Transform
scale_numeric: false       # keep sparse; set true later if RAM allows
ohe_min_freq: 20           # bucket rare categories to shrink OHE width

# Splits
test_size: 0.15
val_size: 0.15
random_state: 42
stratify: true
```

---

## 5) Commands

### Download data from Kaggle

```powershell
make fetch
```

* Downloads & unzips **US_Accidents** into `data/raw/`.
* Update `configs/default.yaml → raw_csv_path` to the printed CSV name (e.g., `US_Accidents_March23.csv`).

### Preprocess (train/val/test + artifacts)

```powershell
make preprocess
```

---

## 6) What You Get After `make preprocess`

**Processed datasets**

```
data/processed/
  X_train.npz          # sparse CSR matrix
  X_val.npz
  X_test.npz
  y_train.csv          # targets
  y_val.csv
  y_test.csv
  pipeline.joblib      # fitted ColumnTransformer
  features.json        # ordered feature names (incl. OHE)
  meta.json            # shapes, paths, timestamps
```

**EDA reports**

```
reports/
  eda_overview.json    # rows/cols, timestamp, columns
  class_distribution.csv
  missingness.csv
  numeric_stats.csv
```

Quick checks:

```powershell
Get-ChildItem data\processed
Get-Content data\processed\meta.json
Get-Content reports\eda_overview.json
```

---

## 7) Loading Artifacts for Modeling

```python
from pathlib import Path
import json, joblib, numpy as np
from scipy.sparse import load_npz

root = Path("data/processed")
X_train = load_npz(root/"X_train.npz")
X_val   = load_npz(root/"X_val.npz")
X_test  = load_npz(root/"X_test.npz")

y_train = np.loadtxt(root/"y_train.csv", delimiter=",")
y_val   = np.loadtxt(root/"y_val.csv", delimiter=",")
y_test  = np.loadtxt(root/"y_test.csv", delimiter=",")

pipeline = joblib.load(root/"pipeline.joblib")
features = json.loads((root/"features.json").read_text())["feature_names"]
```

> Matrices are **sparse** on purpose to avoid huge RAM usage after one-hot.

---

## 8) Typical Workflow

1. `make fetch` → pull latest Kaggle CSV.
2. Edit `configs/default.yaml`:

   * Confirm `raw_csv_path`.
   * Start with `sample_n_rows: 100000` and `ohe_min_freq: 20`.
3. `make preprocess` → generates artifacts.
4. Train a quick baseline on `X_train.npz`/`y_train.csv`.
5. If stable, bump to `sample_n_rows: 200000` → `500000` → all rows (`null`).

---

## 9) Troubleshooting

* **`FileNotFoundError: ... raw CSV`**
  Your `raw_csv_path` doesn’t match the real file. List files:

  ```powershell
  Get-ChildItem data\raw -File *.csv | Sort-Object Length -Descending
  ```

  Paste the correct name into `configs/default.yaml`.

* **MemoryError / huge dense arrays**
  You’re safe now: outputs are `.npz` (sparse). Keep:

  * `scale_numeric: false`
  * `ohe_min_freq: 20`
  * `sample_n_rows: 100000` (raise later)

* **`ModuleNotFoundError: src` when running scripts**
  Always run `make` from the **repo root**. The scripts add `src/` to `sys.path`.

* **PowerShell vs. Bash activation**
  We **don’t** require activation. Commands call `.\.venv\Scripts\python` via Makefile’s `setup`.

---

## 10) Baselines & Evaluation

After preprocessing is complete, you can train and evaluate simple baseline models
to validate the processed artifacts and establish initial performance benchmarks.

### Command
```bash
make baselines
```

### What This Step Does
Running `make baselines` will:
- Load preprocessed datasets from **`data/processed/`**
- Train a set of quick, lightweight models for comparison:
  - **Dummy (Majority Class)** — predicts the most frequent severity class
  - **Logistic Regression (liblinear)** — fast, interpretable linear model
  - **Logistic Regression (saga)** — supports larger sparse datasets
  - **Linear SVM** — strong baseline for linear separation
  - **Random Forest** — simple ensemble tree-based model
- Evaluate each model on the **validation set only** (the **test set remains untouched** for final evaluation)
- Compute and record:
  - Overall **Accuracy**
  - **Macro F1-Score** (equal weight across all classes)
  - **Per-Class Precision, Recall, and F1**
  - A **Normalized Confusion Matrix** (row-wise percentages showing prediction quality per class)
- Record the **split strategy** used (train/val/test ratios, stratified sampling)
  as defined in `configs/default.yaml`.

---

### 📊 Outputs & Where to Find Results

All generated reports and evaluation artifacts are saved under the `reports/` directory:

```
reports/
  baselines/
    majority.json
    logreg_liblinear.json
    logreg_saga.json
    linearsvc.json
    rf.json
  figures/
    cm_majority.png
    cm_logreg_liblinear.png
    cm_logreg_saga.png
    cm_linearsvc.png
    cm_rf.png
  baselines_summary.csv
```

#### 🔹 JSON Reports
Each file in `reports/baselines/` contains:
- Model parameters (e.g., solver, class weight, random seed)
- Accuracy and Macro F1 scores
- Per-class precision, recall, and F1
- The split configuration (train/val/test ratios, stratified flag)
  
These JSON files serve as **machine-readable logs** for your experiments and can be used later for automated performance tracking or plotting.

#### 🔹 Summary CSV
`reports/baselines_summary.csv` provides a compact, sortable table comparing all models by **accuracy** and **macro-F1**.
You can open this file in Excel, Google Sheets, or Pandas for a quick performance overview.

Example:
| model | accuracy | macro_f1 |
|--------|-----------|----------|
| logreg_saga | 0.74 | 0.68 |
| rf | 0.81 | 0.72 |

#### 🔹 Confusion Matrices
Each model’s confusion matrix (normalized) is saved in `reports/figures/` as `cm_<model>.png`.

- Each cell value shows the **percentage of samples per true class** predicted as another class.
- Darker diagonals indicate stronger per-class accuracy.
- This makes it easy to identify which severity levels the model struggles to distinguish (e.g., severe accidents mislabeled as minor).

---

### 🧠 Why This Step Matters
Running baselines helps you:
- Verify that your preprocessing pipeline worked correctly
- Confirm labels and splits are balanced and usable
- Establish a **performance benchmark** before building more complex models (e.g., XGBoost, deep learning)
- Visually inspect model bias or class confusion early on

> **Tip:** Keep the test set untouched until final evaluation — this ensures
> your metrics reflect real-world generalization rather than overfitting to the validation data.

---
