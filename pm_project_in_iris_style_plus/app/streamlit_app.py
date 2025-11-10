import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title='RUL Predictor (Decision Tree)', layout='wide')
st.title('Predictive Maintenance – RUL (Decision Tree)')

model_path = Path('models/dt_reg.joblib')
if not model_path.exists():
    st.warning('Model not found. Please train first: `python scripts/train.py --grid --split timeaware`')
else:
    model = joblib.load(model_path)
    st.sidebar.header('Upload features CSV')
    f = st.sidebar.file_uploader('CSV with feature columns (settings + sensors)', type=['csv'])
    if f is not None:
        X = pd.read_csv(f)
        st.write('Preview:', X.head())
        yhat = model.predict(X)
        st.metric('Mean predicted RUL (cycles)', f'{yhat.mean():.1f}')
        st.line_chart(pd.Series(yhat, name='RUL'))
        threshold = st.sidebar.number_input('Alert threshold (cycles)', value=50, min_value=1)
        alerts = (yhat <= threshold).sum()
        st.info(f'Engines under threshold: {alerts}')
