import numpy as np
import pytest
import torch

from rule4ml.models.metrics import TorchParametricMAPE, evaluate


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


def test_torch_mape_batches():
    mape = TorchParametricMAPE(0, device=torch.device("cpu"))
    mape.update(torch.ones(2, 1), torch.ones(2, 1))
    mape.update(torch.ones(1, 1), torch.full((1, 1), 2.0))

    assert mape.result().item() == pytest.approx(100 / 3)  # 0%, 0%, 100%


def test_keras_r2_batches():
    pytest.importorskip("tensorflow")
    from rule4ml.models.metrics import KerasParametricR2

    y_true = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y_pred = np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)

    metric = KerasParametricR2(0)
    metric.update_state(y_true[:2], y_pred[:2])
    metric.update_state(y_true[2:], y_pred[2:])

    assert float(metric.result()) == pytest.approx(0.6)
