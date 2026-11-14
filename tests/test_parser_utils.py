import pytest

from rule4ml.parsers.utils import fixed_precision_to_bit_width


@pytest.mark.parametrize(
    ("precision", "bits"),
    [
        ("fixed<8, -2>", (8, -2)),
        ("ap_fixed<16,6,AP_RND,AP_SAT>", (16, 6)),
        ("AP_FIXED<18, 8, RND_CONV, SAT_SYM, 1>", (18, 8)),
    ],
)
def test_parse_precision(precision, bits):
    assert fixed_precision_to_bit_width(precision) == bits


@pytest.mark.parametrize(
    "precision",
    [
        "ap_fixed<16,6>junk",
        "ap_fixed<16,6,AP_SAT>",
        "ap_ufixed<8,3>",
    ],
)
def test_parse_invalid_precision(precision):
    with pytest.raises(ValueError):
        fixed_precision_to_bit_width(precision)
