# Predictive Maintenance (FD001) — Iris-Style Notebook

This repo follows the simple style of your **Iris Decision Tree** project: scikit-learn, train/test split, DecisionTree models, confusion matrices, and seaborn/matplotlib visuals.

## Contents
- **data/**: labeled NASA CMAPSS FD001 CSVs
- **notebooks/pm_fd001_in_iris_style.ipynb**: main analysis & modeling
- **scripts/train.py**: trains a DecisionTreeRegressor and saves metrics
- **scripts/predict.py**: batch predictions from CSV

## Quickstart
```bash
pip install -r requirements.txt
python scripts/train.py --data data/FD001_train_labeled.csv
python scripts/predict.py --model models/dt_reg.joblib --data data/FD001_test_labeled.csv --out predictions.csv
```

## Notes
- Features use common informative sensors: s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21 + settings 1–3.
- Classification framing (fail within 30 cycles) is included in the notebook with **DecisionTreeClassifier** and a confusion matrix, to mirror your coding style.

### New Extras Added
- **Seaborn theme** for consistent visuals.
- **Time-aware validation** (engine-wise holdout): `--split timeaware --val_last_engines 20`.
- **GridSearchCV** toggle: `--grid` to tune Decision Tree hyperparameters.
- **Streamlit app**: run `streamlit run app/streamlit_app.py` after training.

### Examples
```bash
# Train with time-aware split + tuning
python scripts/train.py --split timeaware --val_last_engines 20 --grid

# Launch the demo UI
streamlit run app/streamlit_app.py
```
