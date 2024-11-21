import hashlib
import json
import os
import re

import cpuinfo
import numpy as np
import psutil


class IntRange:
    """
    _summary_

    Args:
        min_int (int): _description_
        max_int (int): _description_
        step (int, optional): _description_. Defaults to 1.
    """

    def __init__(self, min_int: int, max_int: int, step: int = 1):
        self.min = min(min_int, max_int)
        self.max = max_int
        self.step = step

    def random_in_range(self, rng=None, size=None, endpoint=True):
        if rng is None:
            rng = np.random.default_rng()

        return rng.choice(self.to_list(endpoint), size=size)

    def to_list(self, endpoint=True):
        return list(range(self.min, self.max + int(endpoint), self.step))


class Power2Range:
    """
    _summary_

    Args:
        min_int (int):
        max_int (int):
    """

    def __init__(self, min_int: int, max_int: int):
        self.min = min(min_int, max_int)
        self.max = max_int

        self.min_exp = max(0, int(np.log2(self.min)))
        self.max_exp = int(np.log2(self.max))

    def random_in_range(self, rng=None, size=None, endpoint=True):
        if rng is None:
            rng = np.random.default_rng()

        rnd_exp = rng.integers(self.min_exp, high=self.max_exp, size=size, endpoint=endpoint)
        return 2**rnd_exp

    def to_list(self, endpoint=True):
        return [2**x for x in range(self.min_exp, self.max_exp + int(endpoint))]


class FloatRange:
    """
    _summary_

    Args:
        min_float (float):
        max_float (float):
    """

    def __init__(self, min_float: float, max_float: float):
        self.min = min(min_float, max_float)
        self.max = max_float

    def random_in_range(self, rng=None, size=None):
        if rng is None:
            rng = np.random.default_rng()

        return rng.uniform(self.min, self.max, size=size)

    def to_list(self, n_samples=50, endpoint=True):
        return np.linspace(self.min, self.max, num=n_samples, endpoint=endpoint)


def data_from_report(path):
    """
    Grabs resource usage and latency info from the synthesis report file (.rpt)
    Might not work for all Vivado versions. (tested on 2019.X)

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """

    with open(path) as file:
        content = file.read()
    content_lines = content.split("\n")

    total_match = re.search(r"\|Total\s+\|(.+)\|", content)
    latency_match = re.search(r"Latency", content)
    clock_match = re.search(r"\|ap_clk\s+\|(.+)\|", content)

    resource_dict = {}
    if total_match:
        numbers_str = total_match.group(1).split("|")
        keys = ["BRAM", "DSP", "FF", "LUT", "URAM"]
        resource_dict = {
            key.lower(): int(num.strip()) if num.strip().isdigit() else 0
            for key, num in zip(keys, numbers_str)
        }

    latency_dict = {}
    if latency_match:
        numbers_line = content.count("\n", 0, latency_match.start())

        numbers_str = []
        while len(numbers_str) < 2:
            numbers_line += 1
            numbers_str = re.findall(r"\d+\.\d+|\d+", content_lines[numbers_line])

        numbers_str = numbers_str[:2]
        keys = ["cycles_min", "cycles_max"]
        latency_dict.update(
            {
                key: (int(num.strip()) if num.strip().isdigit() else float(num.strip()))
                for key, num in zip(keys, numbers_str)
            }
        )

    if clock_match:
        numbers_line = content.count("\n", 0, clock_match.start())

        numbers_str = numbers_str = re.findall(r"\d+\.\d+|\d+", content_lines[numbers_line])[:2]
        keys = ["target_clock", "estimated_clock"]
        latency_dict.update(
            {
                key: (int(num.strip()) if num.strip().isdigit() else float(num.strip()))
                for key, num in zip(keys, numbers_str)
            }
        )

    return resource_dict, latency_dict


def data_from_synthesis(synth_dict: dict, vsynth):
    resource_dict = {}
    res_keys = ["BRAM", "DSP", "FF", "LUT", "URAM"]
    target_report = "VivadoSynthReport" if vsynth else "CSynthesisReport"

    for k1 in res_keys:
        for k2 in synth_dict[target_report]:
            if k2.startswith(k1):
                resource_dict[k1] = str(float(synth_dict[target_report][k2]))

    latency_dict = {}
    if "CSynthesisReport" in synth_dict:
        latency_dict.update(
            {
                "cycles_min": str(float(synth_dict["CSynthesisReport"]["BestLatency"])),
                "cycles_max": str(float(synth_dict["CSynthesisReport"]["WorstLatency"])),
                "target_clock": str(float(synth_dict["CSynthesisReport"]["TargetClockPeriod"])),
                "estimated_clock": str(
                    float(synth_dict["CSynthesisReport"]["EstimatedClockPeriod"])
                ),
            }
        )

    return resource_dict, latency_dict


def model_name_from_config(model, model_config, hls_config):
    """
    _summary_

    Args:
        model (_type_): _description_
        model_config (_type_): _description_
        hls_config (_type_): _description_

    Returns:
        str: _description_
    """

    model_name = ""
    if hasattr(model, "name"):
        model_name += f"{model.name}_"

    for idx, layer in enumerate(model_config):
        class_name = layer["class_name"]
        if class_name not in ["InputLayer", "Flatten", "Activation"]:
            input_shape = np.asarray(layer["input_shape"]).flatten()
            input_size = np.prod([x for x in input_shape if x is not None])
            model_name += f"""{class_name}_{input_size}in_"""

    output_shape = np.asarray(model_config[-1]["output_shape"]).flatten()
    output_size = np.prod([x for x in output_shape if x is not None])
    model_name += f"{output_size}out_"

    hls_model_config = hls_config["Model"]
    model_name += f"""{hls_model_config["Precision"]}_{hls_model_config["ReuseFactor"]}rf_"""
    model_name += f"""{hls_model_config["Strategy"][0].upper()}"""

    return model_name


def print_hls_config(d, indent=0):
    """
    _summary_

    Args:
        d (_type_): _description_
        indent (int, optional): _description_. Defaults to 0.
    """

    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_hls_config(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))


def save_to_json(data, file_path="./dataset.json", indent=2):
    """
    _summary_

    Args:
        data (_type_): _description_
        file_path (str, optional): _description_. Defaults to './dataset.json'.
        indent (int, optional): _description_. Defaults to 2.
    """

    if not os.path.isfile(file_path):  # Create file if it doesn't exist
        with open(file_path, "w") as json_file:
            json.dump([data], json_file, indent=indent)
    else:  # If it exists then just append to the file
        with open(file_path, "r+") as json_file:
            json_file.seek(os.stat(file_path).st_size - 2)
            json_file.write(
                ",\n"
                + " " * indent
                + "{}\n]".format(json.dumps(data, indent=indent).replace("\n", "\n" + " " * indent))
            )


def md5_hash_dict(d):
    """
    _summary_

    Args:
        d (dict): _description_

    Returns:
        _type_: _description_
    """

    dict_string = json.dumps(d, sort_keys=True)
    md5_hash = hashlib.md5(dict_string.encode()).hexdigest()

    return md5_hash


def get_cpu_info():
    """
    _summary_

    Returns:
        _type_: _description_
    """

    all_cpu_info = cpuinfo.get_cpu_info()
    brand = all_cpu_info["brand_raw"].split("@")[0].replace("CPU ", "").strip()
    arch = all_cpu_info["arch"]
    hz_advertised_friendly = all_cpu_info.get("hz_advertised_friendly", "N/A")

    count_logical = psutil.cpu_count(logical=True)
    count_physical = psutil.cpu_count(logical=False)

    return {
        "brand": brand,
        "architecture": arch,
        "base_frequency": hz_advertised_friendly,
        "logical_count": count_logical,
        "physical_count": count_physical,
    }
