from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from rentsense.config import TrainConfig
from rentsense.preprocess import prepare_xy
from rentsense.scoring import score_dataframe, top_examples

st.set_page_config(page_title="RentSense", layout="wide")

st.title("RentSense — прогноз аренды и поиск переоценённых объявлений")
st.caption(
    "Загрузи CSV → получи прогноз и флаги overpriced/underpriced (если есть колонка baseRent)."
)

model_path = st.sidebar.text_input("Путь к модели (.joblib)", value="artifacts/best_model.joblib")
th_pct = st.sidebar.slider(
    "Порог по проценту (±)", min_value=0.05, max_value=0.50, value=0.20, step=0.01
)
th_eur = st.sidebar.slider(
    "Порог по евро (±)", min_value=50.0, max_value=500.0, value=150.0, step=10.0
)
target = st.sidebar.text_input("Target column (если есть)", value="baseRent")

uploaded = st.file_uploader("CSV файл (например, immo_data.csv)", type=["csv"])

if not uploaded:
    st.info("Загрузи CSV, чтобы увидеть прогноз и флаги.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

if not Path(model_path).exists():
    st.error(f"Модель не найдена: {model_path}. Сначала запусти `rentsense train ...`")
    st.stop()

est = joblib.load(model_path)

cfg = TrainConfig(target=target, th_pct=th_pct, th_eur=th_eur)
prepared = prepare_xy(df, cfg)
X = prepared.X

y_true = None
if target in df.columns:
    y_true = pd.to_numeric(df.loc[X.index, target], errors="coerce")

scored = score_dataframe(est, X, y_true=y_true, th_pct=th_pct, th_eur=th_eur)

st.subheader("Результат")
cols = st.columns(3)
cols[0].metric("Rows scored", len(scored))
cols[1].metric("Features", scored.shape[1])
cols[2].metric("Columns dropped", sum(len(v) for v in prepared.dropped.values()))

st.write("Флаги:")
st.dataframe(scored["flag"].value_counts().to_frame("count"), use_container_width=True)

over, under = top_examples(scored, n=10)

tab1, tab2, tab3 = st.tabs(["Scored table", "Top overpriced", "Top underpriced"])
with tab1:
    st.dataframe(scored.head(100), use_container_width=True)
with tab2:
    st.dataframe(over, use_container_width=True)
with tab3:
    st.dataframe(under, use_container_width=True)

csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button("Скачать scored.csv", data=csv_bytes, file_name="scored.csv", mime="text/csv")
