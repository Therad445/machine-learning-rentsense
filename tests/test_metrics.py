from rentsense.metrics import evaluate_regression, mape


def test_mape_basic():
    y_true = [100, 200, 300]
    y_pred = [110, 180, 330]
    val = mape(y_true, y_pred)
    assert 0 <= val < 1


def test_evaluate_regression_keys():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 3, 4]
    d = evaluate_regression("x", y_true, y_pred)
    assert set(d.keys()) == {"model", "MAE", "RMSE", "MAPE", "R2"}
