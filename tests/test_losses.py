import math

import pytest
import torch

from rule4ml.models.losses import get_loss_from_str


def test_tweedie_deviance():
    loss = get_loss_from_str("tweedie", backend="torch", reduction="none")
    values = loss(torch.tensor([2.0, 8.0]), torch.full((2,), 8.0))

    assert values[0].item() == pytest.approx(4 * math.sqrt(2), rel=1e-5)
    assert values[1].item() == pytest.approx(0.0, abs=1e-5)
