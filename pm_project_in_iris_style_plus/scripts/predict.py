# Predict script (simple)
import pandas as pd, joblib, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/dt_reg.joblib')
    ap.add_argument('--data', required=True, help='CSV with same feature columns as training')
    ap.add_argument('--out', default='predictions.csv')
    args = ap.parse_args()

    model = joblib.load(args.model)
    X = pd.read_csv(args.data)
    yhat = model.predict(X)
    pd.DataFrame({'prediction': yhat}).to_csv(args.out, index=False)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
