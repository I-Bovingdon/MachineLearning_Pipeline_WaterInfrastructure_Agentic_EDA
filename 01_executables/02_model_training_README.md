# Part 2 of 3 — Model Training

`02_model_training.ipynb` · Pipeline position: **`01 → [ 02 ] → 03`**

The complete modelling arc: from a raw-data baseline to the final weighted voting ensemble that scored **0.8230** on the DrivenData leaderboard — with a dedicated block of four strategies targeting the operationally critical `functional needs repair` class.

**Prerequisite:** run `01_data_preparation.ipynb` first. This notebook reads the engineered matrices from `artifacts/` and never repeats EDA or feature engineering.

---

## What this notebook does

### 2. Baseline on raw data
A 100-tree Random Forest with only the minimum viable preprocessing (median imputation + LabelEncoder). It establishes the reference point — **0.8076 accuracy** — that quantifies the value of everything that follows.

![Baseline confusion matrix](images/fig_cm_baseline.png)

### 3. Optimised model + interpretability
The same architecture on the engineered features (500 trees, `class_weight='balanced_subsample'`), validated with stratified 5-fold CV, plus the interpretability block: feature importance ranking, cumulative importance curve, and low-importance variable audit.

![Feature importance](images/fig_feature_importance_optimized.png)

![Confusion matrix comparison](images/fig_cm_comparison.png)

### 4–5. Model search
- **AutoML** (`RandomizedSearchCV` over Random Forest, Extra Trees and Gradient Boosting)
- **Stacking Ensemble** — RF + ET + GBT base learners feeding a LogisticRegression meta-learner on out-of-fold probabilities (leakage-free via `cross_val_predict`)

![AutoML comparison](images/fig_automl_comparison.png)

![Stacking comparison](images/fig_stacking_comparison.png)

### 6. Minority class strategies — the key engineering challenge

Standard models detect only **27–31%** of `functional needs repair` pumps. Four strategies attack this, each with an explicit trade-off:

| Strategy | Repair detection | Global accuracy | Verdict |
|---|---|---|---|
| A — Threshold tuning | ~81.5% recall | ~70% | Best detection, too many false positives |
| B — Two-stage model | Moderate | Minimal loss | Balanced, two classifiers |
| C — Cost-sensitive learning | Moderate | Minimal loss | Single model, penalty-weighted |
| D — **Weighted voting ensemble** | 6.7% predicted (vs 7.3% real) | **0.8230** | **Final solution** |

![Threshold tuning](images/fig_threshold_tuning.png)

![Repair recall comparison](images/fig_repair_recall_comparison.png)

![Ensemble voting](images/fig_ensemble_votacion.png)

### 7–9. Final model, submissions and summary
The final Random Forest is retrained on 100% of the training data, all `submission_*.csv` files are generated, and the unified metric table (accuracy · macro-F1 · repair-recall) closes the analysis.

![Model comparison](images/fig_model_comparison.png)

![Process summary](images/fig_summary.png)

---

## Model performance

| Model | Accuracy | Notes |
|---|---|---|
| Random Forest Baseline | 0.8076 | Raw data |
| Optimised Random Forest | 0.8102 | Feature engineering |
| AutoML (RandomizedSearchCV) | 0.8104 | Hyperparameter search |
| Stacking Ensemble | 0.8114 | RF + ET + GBT meta-learner |
| **Weighted Voting Ensemble** | **0.8230** | **Final — best repair detection** |

## Outputs

```
submission_modelo_base.csv          # baseline reference
submission_pump_it_up.csv           # optimised RF
submission_stacking.csv             # stacking ensemble
submission_threshold.csv            # strategy A
submission_dos_etapas.csv           # strategy B
submission_cost_sensitive.csv       # strategy C
submission_ensemble_votacion.csv    # strategy D — final (0.8230)
artifacts/
├── rf_final.joblib                 # final model            → 03
└── model_config.json               # tuned BEST_THR          → 03
```

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow joblib
```

1. Run `01_data_preparation.ipynb` first (produces `artifacts/`)
2. `Kernel → Restart & Run All`

> ⏱ The AutoML search (section 4) and stacking CV (section 5) are the heaviest cells: expect 15–40 minutes total depending on CPU cores. All estimators use `n_jobs=-1`.

Next: **[`03_interactive_annexes.ipynb`](03_interactive_annexes_README.md)**
