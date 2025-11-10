# Train script with GridSearch and time-aware split
import pandas as pd, numpy as np, json, joblib, argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

def time_aware_engine_split(df, feat_cols, target_col='RUL', val_last_engines=20):
    engine_ids = sorted(df['unit'].unique())
    val_ids = set(engine_ids[-val_last_engines:])
    tr_df = df[~df['unit'].isin(val_ids)]
    va_df = df[df['unit'].isin(val_ids)]
    Xtr, ytr = tr_df[feat_cols].copy(), tr_df[target_col].copy()
    Xva, yva = va_df[feat_cols].copy(), va_df[target_col].copy()
    return Xtr, Xva, ytr, yva

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/FD001_train_labeled.csv')
    ap.add_argument('--out', default='models/dt_reg.joblib')
    ap.add_argument('--split', default='random', choices=['random','timeaware'])
    ap.add_argument('--val_last_engines', type=int, default=20)
    ap.add_argument('--grid', action='store_true', help='Enable GridSearchCV for tuning')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    keep_sensors = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
    feat_cols = ['setting1','setting2','setting3'] + keep_sensors

    X = df[feat_cols]; y = df['RUL']

    if args.split == 'timeaware':
        Xtr, Xva, ytr, yva = time_aware_engine_split(df, feat_cols, target_col='RUL', val_last_engines=args.val_last_engines)
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42, max_depth=10)
    if args.grid:
        param_grid = {'max_depth':[5,10,15,None], 'min_samples_leaf':[1,5,10]}
        gs = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
        gs.fit(Xtr, ytr)
        model = gs.best_estimator_
    else:
        model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    rmse = float(np.sqrt(((yva - pred)**2).mean()))
    mae = float(np.abs(yva - pred).mean())
    r2 = float(r2_score(yva, pred))

    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, args.out)
    with open('models/metrics.json','w') as f:
        json.dump({'rmse': rmse, 'mae': mae, 'r2': r2}, f, indent=2)
    print('Saved', args.out, {'rmse': rmse, 'mae': mae, 'r2': r2})

if __name__ == '__main__':
    main()
