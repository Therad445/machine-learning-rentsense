# RentSense

RentSense — прогноз “справедливой” аренды и поиск переоценённых/недооценённых объявлений.

Что делает проект:
- Загружает `./data/immo_data.csv`
- Делает краткий профиль + EDA и сохраняет картинки в `./artifacts/`
- Обучает 2 модели: Ridge (baseline) + HistGBR (strong) с log1p-трансформацией таргета
- Считает метрики MAE/RMSE/MAPE/R², а также метрики по ценовым сегментам
- Строит permutation importance
- Помечает объявления как `overpriced/underpriced/ok` по правилам (процент + евро)
- Сохраняет артефакты: модель, таблицы, графики, примеры

## Быстрый старт

1) Положи датасет:
```
data/immo_data.csv
```

2) Установи зависимости:
```
python -m venv .venv
# windows: .venv\Scripts\activate
# linux/mac: source .venv/bin/activate
pip install -e .
```

3) Обучение и артефакты:
```
rentsense train --data data/immo_data.csv --artifacts artifacts --split random --fast
```

4) Скоринг новых объявлений:
```
rentsense score --model artifacts/best_model.joblib --input data/immo_data.csv --out artifacts/scored.csv
```

## Streamlit демо
```
streamlit run app/streamlit_app.py
```

## API + метрики (SRE-обвязка)
```
uvicorn service.api:app --host 0.0.0.0 --port 8000
# /health
# /score
# /metrics  (Prometheus)
```

## Docker
```
docker compose up --build
```

## Артефакты
После `train` в `artifacts/` появятся:
- `metrics.csv`
- `metrics_by_segment.csv`
- `missing_top30.csv`
- `eda_*.png`
- `feature_importance_top20.csv/png`
- `examples_overpriced_top10.csv`
- `examples_underpriced_top10.csv`
- `pred_vs_true.png`
- `best_model.joblib`
- `run_metadata.json`
