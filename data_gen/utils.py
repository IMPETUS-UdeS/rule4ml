import hashlib
import json
import os
import re

import cpuinfo
import numpy as np
import psutil


def data_from_report(path):
    """
    Grabs resource usage and latency info from the synthesis report file (.rpt)
    Might not work for all Vivado versions. (tested on 2019.X)
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

        numbers_str = re.findall(r"\d+\.\d+|\d+", content_lines[numbers_line])[:2]
        keys = ["target_clock", "estimated_clock"]
        latency_dict.update(
            {
                key: (int(num.strip()) if num.strip().isdigit() else float(num.strip()))
                for key, num in zip(keys, numbers_str)
            }
        )

    return resource_dict, latency_dict


def data_from_synthesis(synth_dict: dict):
    resource_dict = {}
    res_keys = ["BRAM", "DSP", "FF", "LUT", "URAM"]
    target_reports = ["CSynthesisReport", "VivadoSynthReport"]

    for report in target_reports:
        if report in synth_dict:
            resource_dict[report] = {}
            for k1 in res_keys:
                for k2 in synth_dict[report]:
                    if k2.startswith(k1):
                        resource_dict[report][k1] = str(float(synth_dict[report][k2]))

    latency_dict = {}
    report = "CSynthesisReport"
    if report in synth_dict:
        latency_dict.update(
            {
                "cycles_min": str(float(synth_dict[report]["BestLatency"])),
                "cycles_max": str(float(synth_dict[report]["WorstLatency"])),
                "target_clock": str(float(synth_dict[report]["TargetClockPeriod"])),
                "estimated_clock": str(
                    float(synth_dict[report]["EstimatedClockPeriod"])
                ),
            }
        )

    return resource_dict, latency_dict


def model_name_from_config(model, model_config, hls_config):
    """Build a descriptive model name from layer sizes and the HLS config."""

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
    model_name += (
        f"""{hls_model_config["Precision"]}_{hls_model_config["ReuseFactor"]}rf_"""
    )
    model_name += f"""{hls_model_config["Strategy"][0].upper()}"""

    return model_name


def print_hls_config(d, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_hls_config(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))


def save_to_json(data, file_path="./dataset.json", indent=2):
    """Append a record to a JSON list file, creating the file if needed."""

    if not os.path.isfile(file_path):
        with open(file_path, "w") as json_file:
            json.dump([data], json_file, indent=indent)
    else:
        with open(file_path, "r+") as json_file:
            json_file.seek(os.stat(file_path).st_size - 2)
            json_file.write(
                ",\n"
                + " " * indent
                + "{}\n]".format(
                    json.dumps(data, indent=indent).replace("\n", "\n" + " " * indent)
                )
            )


def md5_hash_dict(d):
    dict_string = json.dumps(d, sort_keys=True)
    md5_hash = hashlib.md5(dict_string.encode()).hexdigest()

    return md5_hash


def get_cpu_info():
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
