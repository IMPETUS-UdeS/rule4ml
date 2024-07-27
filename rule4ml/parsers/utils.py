import math
import re
from bisect import bisect_left


def camel_keys_to_snake(d: dict, recursive=True):
    """
    Transforms dictionary keys from CamelCase to snake_case format.

    Args:
        d (dict): Targeted dictionary for keys conversion.
        recursive (bool, optional): Whether to apply to nested dictionaries. Defaults to True.

    Returns:
        dict: Transformed dictionary.
    """

    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        new_key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
        if recursive and isinstance(value, dict):
            new_dict[new_key] = camel_keys_to_snake(value, recursive=True)
        else:
            new_dict[new_key] = value

    return new_dict


def unwrap_nested_dicts(d, prefix=None):
    """
    _summary_

    Args:
        d (_type_): _description_
        prefix (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if not isinstance(d, dict):
        return d

    items = []
    for key, value in d.items():
        new_key = f"{prefix}_{key}" if prefix is not None else key
        if isinstance(value, dict):
            items += unwrap_nested_dicts(value, prefix=new_key).items()
        else:
            items.append((new_key, value))

    return dict(items)


def fixed_precision_to_bit_width(precision: str):
    """
    _summary_

    Args:
        precision (str): _description_

    Returns:
        Tuple[int, int]: _description_
    """

    match = re.match(r"ap_fixed<(\d+),\s*(\d+)>", precision.lower())
    if not match:
        raise ValueError(f"Invalid format: {precision}, expecting \"ap_fixed<a, b>\"")

    total_bits, fractional_bits = map(int, match.groups())
    return total_bits, fractional_bits


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L198
def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L213
def get_closest_reuse_factor(n_in, n_out, chosen_rf):
    """
    Returns closest value to chosen_rf.
    If two numbers are equally close, return the smallest number.
    """

    max_rf = n_in * n_out
    valid_reuse_factors = []
    for rf in range(1, max_rf + 1):
        _assert = _validate_reuse_factor(n_in, n_out, rf)
        if _assert:
            valid_reuse_factors.append(rf)
    valid_rf = sorted(valid_reuse_factors)

    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]
    before = valid_rf[pos - 1]
    after = valid_rf[pos]
    if after - chosen_rf < chosen_rf - before:
        return after
    else:
        return before
