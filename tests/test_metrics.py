import numpy as np
import pytest

from rule4ml.models.metrics import evaluate


def test_evaluate_partial_targets():
    y_true = np.array([[1.0, 4.0], [2.0, np.nan], [3.0, 4.0]])
    y_pred = np.array([[3.0, 4.0], [2.0, 9.0], [3.0, 5.0]])

    scores = evaluate(y_true, y_pred, ["lut", "cycles"])

    assert scores["lut"]["n"] == 3
    assert scores["lut"]["smape"] == pytest.approx(80 / 3)  # 200 * 2 / 5, over 3
    assert scores["lut"]["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert scores["lut"]["r2"] == pytest.approx(1 - 4 / 2)
    assert scores["cycles"]["n"] == 2
    assert scores["cycles"]["n_missing"] == 1
    assert scores["cycles"]["smape"] == pytest.approx(100 * 1 / 10)
    assert np.isnan(scores["cycles"]["r2"])  # constant targets


def test_evaluate_nonfinite():
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(np.ones((2, 1)), np.array([[1.0], [np.nan]]), ["lut"])


@pytest.mark.parametrize("prediction", [np.ones(2), np.ones((1, 1)), np.ones((2, 2))])
def test_evaluate_shape_mismatch(prediction):
    with pytest.raises(ValueError, match="shape"):
        evaluate(np.ones((2, 1)), prediction, ["lut"])
