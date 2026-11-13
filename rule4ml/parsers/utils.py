import json
import math
import os
import re
from bisect import bisect_left

# hls4ml fixed-point grammar: width, integer bits, rounding, saturation, sat bits
_ROUNDING = r"(?:ap_)?(?:trn|trn_zero|rnd|rnd_zero|rnd_inf|rnd_min_inf|rnd_conv)"
_SATURATION = r"(?:ap_)?(?:wrap|sat|sat_zero|sat_sym)"
_FIXED_PRECISION = re.compile(
    rf"(?:ap_)?fixed<(\d+),(-?\d+)(?:,{_ROUNDING}(?:,{_SATURATION}(?:,\d+)?)?)?>",
    re.IGNORECASE,
)


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


def to_lower_keys(d: dict, recursive=True):
    """
    Transforms dictionary keys to lower case.

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
        new_key = key.lower()
        if recursive and isinstance(value, dict):
            new_dict[new_key] = to_lower_keys(value, recursive=True)
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


def get_activation_name(activation):
    match = re.match(r"quantized_(\w+)(?:\(?[^)]*\)?)", activation)
    if match:
        return match.group(1)
    return activation


def fixed_precision_to_bit_width(precision: str):
    """Return (total_bits, integer_bits) of an hls4ml ap_fixed precision string."""
    match = _FIXED_PRECISION.fullmatch(re.sub(r"\s", "", precision))
    if not match:
        raise ValueError(
            f'Invalid precision: {precision}, expecting "ap_fixed<W,I[,Q[,O[,N]]]>"'
        )
    return int(match[1]), int(match[2])


def get_supported_boards():
    """
    _summary_

    Returns:
        _type_: _description_
    """

    with open(
        os.path.join(os.path.dirname(__file__), "supported_boards.json")
    ) as json_file:
        supported_boards = json.load(json_file)

    return supported_boards


def get_board_from_part(part_number):
    """
    _summary_

    Args:
        part_number (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    supported_boards = get_supported_boards()

    for key, value in supported_boards.items():
        if value["part"] == part_number:
            return key

    raise ValueError(f"Part number {part_number} not supported.")


def get_part_from_board(board_name):
    """
    _summary_

    Args:
        board_name (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    supported_boards = get_supported_boards()

    if board_name not in supported_boards:
        raise ValueError(f"Board {board_name} not supported.")

    return supported_boards[board_name]["part"]


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L262
def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L277
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
