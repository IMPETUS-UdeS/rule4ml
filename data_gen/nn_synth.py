import fnmatch
import inspect
import json
import os
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime

import hls4ml
import numpy as np
import tensorflow as tf
from hls4ml.backends.backend import get_available_backends, get_backend
from nn_gen import GeneratorSettings, generate_fc_network
from utils import (
    IntRange,
    Power2Range,
    data_from_synthesis,
    get_cpu_info,
    md5_hash_dict,
    model_name_from_config,
    print_hls_config,
    save_to_json,
)

from rule4ml.parsers.network_parser import (
    config_from_keras_model,
    config_from_torch_model,
    keras_model_from_config,
)
from rule4ml.parsers.utils import get_board_from_part, get_part_from_board

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dynamically get the arguments for each hls4ml backend build method
backend_build_args = {
    backend_name: {
        arg.name: arg.default if arg.default is not inspect._empty else None
        for arg in list(inspect.signature(get_backend(backend_name).build).parameters.values())[1:]
    }
    for backend_name in get_available_backends()
}

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set Xilinx license file if available
if os.path.exists("secrets.json"):
    with open("secrets.json") as f:
        secrets = json.load(f)
        if "XILINXD_LICENSE_FILE" in secrets:
            os.environ["XILINXD_LICENSE_FILE"] = secrets["XILINXD_LICENSE_FILE"]


@dataclass
class SynthSettings:
    """
    Data class that holds parameter ranges for NN synthesis.
    Helper Range classes are used for easy random (uniform) sampling.

    Args: (all optional, default values inside data class)
        reuse_range (Power2Range): Reuse factor range.
        precisions (list): List of synthesis precisions.
        strategies (list): List of synthesis strategies (Latency, Resource).
        boards (list): List of synthesis boards.
        clock_period_range (IntRange): Clock period range.
        io_types (list): List of io types.
    """

    # Default settings, used for original dataset

    reuse_range: Power2Range = Power2Range(1, 64)
    precisions: list = field(
        default_factory=lambda: [
            "ap_fixed<2, 1>",
            "ap_fixed<8, 3>",
            "ap_fixed<8, 4>",
            "ap_fixed<16, 6>",
        ]
    )
    strategies: list = field(default_factory=lambda: ["Latency", "Resource"])
    boards: list = field(default_factory=lambda: ["pynq-z2", "zcu102", "alveo-u200"])

    clock_period_range: IntRange = IntRange(10, 10)
    io_types: list = field(default_factory=lambda: ["io_parallel"])

    backends: list = field(default_factory=lambda: ["VivadoAccelerator"])
    backend_versions: list = field(default_factory=lambda: ["2019.1"])


def keras_to_hls4ml(
    model,
    project_dir,
    hls_config,
    clock_period,
    part_number,
    io_type,
    backend,
    verbose=0,
):
    """
    Creates an hls4ml Model using a keras Model and hls4ml settings.

    Args:
        model (keras.Model): The Keras model to convert.
        project_dir (str): The output directory for the hls4ml project.
        hls_config (dict): A dict that contains nested config dicts, matching the hls4ml generated config.
            The two top-level keys should be "Model" and either of "LayerType" or "LayerName".
        clock_period (str): The target clock period for the synthesis.
        part_number (str): The part number of the FPGA board.
        io_type (str): The io type for the synthesis. Should be either "io_parallel" or "io_stream".
        backend (str): The backend to use for the synthesis. Check hls4ml documentation for available backends.
        verbose (int, optional): Output verbosity. Defaults to 0.

    Returns:
        hls4ml.Model: The created hls4ml model.
    """

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=project_dir,
        backend=backend,
        clock_period=clock_period,
        io_type=io_type,
        board=get_board_from_part(part_number),
    )
    hls_model.compile()

    if verbose > 0:
        print("-----------------------------------")
        print("Configuration")
        print_hls_config(hls_config)
        print("-----------------------------------")

    return hls_model


def torch_to_hls4ml(
    model,
    input_shape,
    project_dir,
    hls_config,
    clock_period,
    part_number,
    io_type,
    backend,
    verbose=0,
):
    """
    Creates an hls4ml Model using a Pytorch Model and hls4ml settings.

    Args:
        model (torch.nn.Module): The Pytorch model to convert.
        input_shape (tuple): The input shape of the model.
        project_dir (str): The output directory for the hls4ml project.
        hls_config (dict): A dict that contains nested config dicts, matching the hls4ml generated config.
            The two top-level keys should be "Model" and either of "LayerType" or "LayerName".
        clock_period (str): The target clock period for the synthesis.
        part_number (str): The part number of the FPGA board.
        io_type (str): The io type for the synthesis. Should be either "io_parallel" or "io_stream".
        backend (str): The backend to use for the synthesis. Check hls4ml documentation for available backends.
        verbose (int, optional): Output verbosity. Defaults to 0.

    Returns:
        hls4ml.Model: The created hls4ml model.
    """

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape,
        hls_config=hls_config,
        output_dir=project_dir,
        backend=backend,
        clock_period=clock_period,
        io_type=io_type,
        board=get_board_from_part(part_number),
    )
    hls_model.compile()

    if verbose > 0:
        print("-----------------------------------")
        print("Configuration")
        print_hls_config(hls_config)
        print("-----------------------------------")

    return hls_model


def add_backend_to_env(backend, backend_version, xilinx_path):
    available_backends = get_available_backends()
    if backend.lower() not in available_backends:
        raise ValueError(f"Backend {backend} not available. Choose from {available_backends}")

    backend_frmt = backend[0].upper() + backend[1:].lower()
    if backend_frmt.endswith("accelerator"):
        backend_frmt = backend_frmt.replace("accelerator", "")

    backend_path = os.path.join(xilinx_path, backend_frmt, backend_version, "bin")
    if not os.path.exists(backend_path):
        raise Exception(
            f"{backend_frmt} {backend_version} installation not found at \"{backend_path}\". "
            + "Please check if the path points to the chosen backend."
        )

    if backend_path not in os.environ["PATH"]:
        os.environ["PATH"] = f"{backend_path}:" + os.environ["PATH"]


def synthesize_keras_model(
    model,
    output_dir,
    hls_config,
    clock_period,
    part_number,
    io_type,
    backend,
    backend_version,
    xilinx_path,
    model_id=None,
    model_name="",
    verbose=0,
    **kwargs,
):
    """
    Synthesize a given Keras model with a specific config, returning the synthesis results.

    Args:
        model (keras.Model): The Keras model to synthesize.
        output_dir (_type_): _description_
        hls_config (_type_): _description_
        clock_period (_type_): _description_
        part_number (_type_): _description_
        io_type (_type_): _description_
        backend (_type_): _description_
        backend_version (_type_): _description_
        xilinx_path (str, optional): _description_. Defaults to "/opt/Xilinx".
        model_id (_type_, optional): _description_. Defaults to None.
        model_name (str, optional): _description_. Defaults to "".
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        dict: _description_
    """

    add_backend_to_env(backend, backend_version, xilinx_path)

    model_config = config_from_keras_model(model, hls_config)

    updated_model_id = md5_hash_dict({"model_config": model_config, "hls_config": hls_config})
    if model_id is None:
        model_id = updated_model_id

    if not model_name:
        model_name = model_name_from_config(model, model_config, hls_config)

    meta_data = {
        "model_id": str(updated_model_id),
        "model_name": model_name,
        "artifacts_file": f"{updated_model_id}.tar.gz",
    }

    project_dir = os.path.join(output_dir, "projects", str(model_id))
    hls_model = keras_to_hls4ml(
        model,
        project_dir,
        hls_config,
        clock_period,
        part_number,
        io_type,
        backend.lower(),
        verbose=verbose,
    )

    build_kwargs = backend_build_args[backend.lower()]
    for key, value in kwargs.items():
        if key in build_kwargs:
            build_kwargs[key] = value

    if verbose > 0:
        print(f"Build args: {build_kwargs}")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    synth_result = hls_model.build(**build_kwargs)
    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    meta_data["synthesis_info"] = [
        {
            "start_time": start_time,
            "end_time": end_time,
            "cpu": get_cpu_info(),
            "build_args": build_kwargs,
        }
    ]

    resource_report, latency_report = data_from_synthesis(synth_result)

    del model
    del hls_model
    tf.keras.backend.clear_session()

    return {
        "meta_data": meta_data,
        "model_config": model_config,
        "hls_config": hls_config,
        "resource_report": resource_report,
        "latency_report": latency_report,
        "target_part": part_number,
        "io_type": io_type,
        "backend": backend,
        "backend_version": backend_version,
        "hls4ml_version": hls4ml.__version__,
    }


def synthesize_torch_model(
    model,
    output_dir,
    hls_config,
    clock_period,
    part_number,
    io_type,
    backend,
    backend_version,
    xilinx_path,
    model_id=None,
    model_name="",
    verbose=0,
    **kwargs,
):
    """
    Synthesize a given Pytorch model with a specific config, returning the synthesis results.

    Args:
        model (torch.nn.Module): The Pytorch model to synthesize.
        output_dir (_type_): _description_
        hls_config (_type_): _description_
        clock_period (_type_): _description_
        part_number (_type_): _description_
        io_type (_type_): _description_
        backend (_type_): _description_
        backend_version (_type_): _description_
        xilinx_path (str, optional): _description_. Defaults to "/opt/Xilinx".
        model_id (_type_, optional): _description_. Defaults to None.
        model_name (str, optional): _description_. Defaults to "".
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        dict: _description_
    """

    add_backend_to_env(backend, backend_version, xilinx_path)

    model_config = config_from_torch_model(model, hls_config)
    input_shape = model_config[0]["input_shape"]

    updated_model_id = md5_hash_dict({"model_config": model_config, "hls_config": hls_config})
    if model_id is None:
        model_id = updated_model_id

    if not model_name:
        model_name = model_name_from_config(model, model_config, hls_config)

    meta_data = {
        "model_id": str(updated_model_id),
        "model_name": model_name,
        "artifacts_file": f"{updated_model_id}.tar.gz",
    }

    project_dir = os.path.join(output_dir, "projects", str(model_id))
    hls_model = torch_to_hls4ml(
        model,
        input_shape,
        project_dir,
        hls_config,
        clock_period,
        part_number,
        io_type,
        backend.lower(),
        verbose=verbose,
    )

    build_kwargs = backend_build_args[backend.lower()]
    for key, value in kwargs.items():
        if key in build_kwargs:
            build_kwargs[key] = value

    if verbose > 0:
        print(f"Build args: {build_kwargs}")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    synth_result = hls_model.build(**build_kwargs)
    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    meta_data["synthesis_info"] = [
        {
            "start_time": start_time,
            "end_time": end_time,
            "cpu": get_cpu_info(),
            "build_args": build_kwargs,
        }
    ]

    resource_report, latency_report = data_from_synthesis(synth_result)

    del model
    del hls_model

    return {
        "meta_data": meta_data,
        "model_config": model_config,
        "hls_config": hls_config,
        "resource_report": resource_report,
        "latency_report": latency_report,
        "target_part": part_number,
        "io_type": io_type,
        "backend": backend,
        "backend_version": backend_version,
        "hls4ml_version": hls4ml.__version__,
    }


def parallel_synthesis(args):
    """
    _summary_

    Args:
        args (dict): _description_
    """

    output_dir = args.get("output_dir", os.path.join(base_path, "datasets"))

    meta_data = args.get("meta_data", None)
    models = args["models"]
    hls_configs = args["hls_configs"]

    if meta_data and len(meta_data) != len(models) * len(hls_configs):
        raise ValueError(
            "Number of meta_data should be equal to the number of models * hls_configs"
        )

    model_ids = args.get("model_ids", None)
    if model_ids and len(model_ids) != len(models) * len(hls_configs):
        raise ValueError(
            "Number of model_ids should be equal to the number of models * hls_configs"
        )

    parts = args["parts"]
    clock_periods = args["clock_periods"]
    io_types = args["io_types"]

    backends = args["backends"]
    backend_versions = args["backend_versions"]
    xilinx_path = args.get("xilinx_path", "/opt/Xilinx")
    build_args = args["build_args"]

    synth_function = args.get("synth_function", synthesize_keras_model)
    synth_verbose = args.get("synth_verbose", 0)

    prj_compress = args.get("prj_compress", True)
    json_filename = args.get("json_filename", "")

    if len(backends) != len(backend_versions):
        if len(backends) < len(backend_versions):
            backends += [backends[-1]] * (len(backend_versions) - len(backends))
        else:
            backend_versions += [backend_versions[-1]] * (len(backends) - len(backend_versions))

    for model_idx, model in enumerate(models):
        for hls_idx, hls_config in enumerate(hls_configs):
            meta_entry = None
            if meta_data:
                meta_entry = meta_data[model_idx * len(hls_configs) + hls_idx]

            model_id = None
            if model_ids:
                model_id = model_ids[model_idx * len(hls_configs) + hls_idx]

            for part_number in parts:
                for clock_period in clock_periods:
                    for io_type in io_types:
                        for backend, backend_version in zip(backends, backend_versions):
                            try:
                                synth_result = synth_function(
                                    model,
                                    output_dir,
                                    hls_config,
                                    clock_period,
                                    part_number,
                                    io_type,
                                    backend,
                                    backend_version,
                                    xilinx_path=xilinx_path,
                                    model_id=model_id,
                                    verbose=synth_verbose,
                                    **build_args,
                                )

                                if meta_entry:
                                    meta_entry["synthesis_info"] += synth_result["meta_data"][
                                        "synthesis_info"
                                    ]
                                    synth_result["meta_data"]["synthesis_info"] = meta_entry[
                                        "synthesis_info"
                                    ]

                                updated_model_id = synth_result["meta_data"]["model_id"]
                                if model_id is None:
                                    model_id = updated_model_id

                                json_file = json_filename if json_filename else f"{model_id}.json"
                                json_path = os.path.join(output_dir, json_file)

                                if os.path.isfile(json_path):
                                    os.remove(json_path)
                                save_to_json(
                                    synth_result,
                                    os.path.join(output_dir, f"{updated_model_id}.json"),
                                    indent=2,
                                )

                                project_dir = os.path.join(output_dir, "projects")
                                if model_id != updated_model_id:
                                    os.rename(
                                        os.path.join(project_dir, model_id),
                                        os.path.join(project_dir, updated_model_id),
                                    )

                                if prj_compress:
                                    os.remove(
                                        os.path.join(project_dir, f"{model_id}.tar.gz")
                                    )  # remove the tar archive that hls4ml produces

                                    with tarfile.open(
                                        os.path.join(project_dir, f"{updated_model_id}.tar.gz"),
                                        "w:gz",
                                    ) as tar:
                                        tar.add(
                                            os.path.join(project_dir, str(updated_model_id)),
                                            arcname=str(updated_model_id),
                                        )

                                    shutil.rmtree(os.path.join(project_dir, str(updated_model_id)))

                            except Exception as e:
                                print(e)


def parallel_rnd_synthesis(args):
    """
    _summary_

    Args:
        args (dict): _description_
    """

    job_id = args["job_id"]
    args["json_filename"] = f"dataset-{job_id}.json"

    n_models = args.get("n_models", 0)

    rng_seed = args.get("rng_seed", None)
    rng = np.random.default_rng(rng_seed)

    gen_function = args.get("gen_function", generate_fc_network)
    gen_settings = args.get("gen_settings", GeneratorSettings())
    gen_verbose = args.get("gen_verbose", 0)

    synth_settings = args.get("synth_settings", SynthSettings())

    granularity = args.get("granularity", "model")
    if granularity not in ["model", "name", "type"]:
        raise ValueError("Granularity should be one of 'model', 'name', 'type'")

    idx = 0
    while True:
        try:
            gen_network = gen_function(gen_settings, rng=rng, verbose=gen_verbose)
            hls_config = hls4ml.utils.config_from_keras_model(gen_network, granularity=granularity)

            model_precision = rng.choice(synth_settings.precisions)
            model_reuse_factor = int(synth_settings.reuse_range.random_in_range(rng))
            model_strategy = rng.choice(synth_settings.strategies)

            hls_config["Model"]["Precision"] = model_precision
            hls_config["Model"]["ReuseFactor"] = model_reuse_factor
            hls_config["Model"]["Strategy"] = model_strategy

            if granularity == "name":
                for layer_name in hls_config["LayerName"]:
                    layer_dict = {}
                    if isinstance(hls_config["LayerName"][layer_name]["Precision"], dict):
                        for prec_key in hls_config["LayerName"][layer_name]["Precision"]:
                            layer_dict["Precision"][prec_key] = rng.choice(
                                synth_settings.precisions
                            )
                    else:
                        layer_dict["Precision"] = rng.choice(synth_settings.precisions)

                    layer_dict["ReuseFactor"] = int(synth_settings.reuse_range.random_in_range(rng))
                    layer_dict["Strategy"] = rng.choice(synth_settings.strategies)

                    hls_config["LayerName"][layer_name] = layer_dict
            elif granularity == "type":
                layer_type_dict = {}
                for layer_type in hls_config["LayerType"]:
                    if isinstance(hls_config["LayerType"][layer_type]["Precision"], dict):
                        for prec_key in hls_config["LayerType"][layer_type]["Precision"]:
                            layer_type_dict["Precision"][prec_key] = rng.choice(
                                synth_settings.precisions
                            )
                    else:
                        layer_type_dict["Precision"] = rng.choice(synth_settings.precisions)

                    layer_type_dict["ReuseFactor"] = int(
                        synth_settings.reuse_range.random_in_range(rng)
                    )
                    layer_type_dict["Strategy"] = rng.choice(synth_settings.strategies)

                    hls_config["LayerType"][layer_type] = layer_type_dict

            clock_period = str(synth_settings.clock_period_range.random_in_range(rng))
            io_type = rng.choice(synth_settings.io_types)
            board = rng.choice(synth_settings.boards)
            part_number = get_part_from_board(board)

            backend = rng.choice(synth_settings.backends)
            backend_version = rng.choice(synth_settings.backend_versions)

            model_args = args.copy()

            model_args["models"] = [gen_network]
            model_args["hls_configs"] = [hls_config]
            model_args["parts"] = [part_number]
            model_args["clock_periods"] = [clock_period]
            model_args["io_types"] = [io_type]

            model_args["backends"] = [backend]
            model_args["backend_versions"] = [backend_version]

            parallel_synthesis(model_args)

            idx += 1
            if n_models > 0 and idx >= n_models:
                break
        except Exception as e:
            print(e)


def synth_from_json(args):
    json_data = args["json_data"]

    progress_counter = args.get("progress_counter", None)
    progress_lock = args.get("progress_lock", None)

    prj_decompress = args.get("prj_decompress", False)

    for model_entry in json_data:
        meta_data = model_entry["meta_data"]
        model_config = model_entry["model_config"]
        hls_config = model_entry["hls_config"]
        model_id = meta_data.get(
            "model_id",
            md5_hash_dict({"model_config": model_config, "hls_config": hls_config}),
        )

        if prj_decompress:
            project_dir = os.path.join(args["output_dir"], "projects")
            tar_path = os.path.join(args["output_dir"], "projects", f"{model_id}.tar.gz")

            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(project_dir)

                    top_level_dirs = {
                        member.name.split("/")[0]
                        for member in tar.getmembers()
                        if member.name.count("/") > 0
                    }
                    if len(top_level_dirs) == 1:
                        extracted_dir = os.path.join(project_dir, next(iter(top_level_dirs)))
                    else:
                        raise ValueError(f"Unexpected directory structure: {top_level_dirs}")

                    # rename the extracted project folder to the model_id
                    new_dir = os.path.join(project_dir, str(model_id))
                    if extracted_dir != new_dir:
                        os.rename(extracted_dir, new_dir)
            except Exception as e:
                print(e)
                continue

        model = keras_model_from_config(model_config)

        clock_period = str(int(float(model_entry["hls_config"]["clock_period"])))

        if "target_part" in model_entry:
            part_number = model_entry["target_part"]
        elif "board" in hls_config:
            part_number = get_part_from_board(hls_config["board"])
        else:
            raise ValueError("Failed to find target part number")

        io_type = hls_config.get("io_type", "io_parallel")
        backend = model_entry.get("backend", "VivadoAccelerator")
        backend_version = model_entry.get("backend_version", "2019.1")

        args["meta_data"] = [meta_data]
        args["models"] = [model]
        args["hls_configs"] = [hls_config]
        args["model_ids"] = [model_id]
        args["parts"] = [part_number]
        args["clock_periods"] = [clock_period]
        args["io_types"] = [io_type]
        args["backends"] = [backend]
        args["backend_versions"] = [backend_version]

        parallel_synthesis(args)

        if progress_counter is not None and progress_lock is not None:
            with progress_lock:
                progress_counter.value += 1


def sanitize_csynth_data(args):
    data = args["json_data"]
    output_dir = args["output_dir"]
    save_files = args.get("save_files", False)

    progress_counter = args.get("progress_counter", None)
    progress_lock = args.get("progress_lock", None)

    sanitized_data = []
    for entry in data:
        sanitized_entry = {}
        sanitized_entry["meta_data"] = {}
        sanitized_entry["meta_data"]["model_id"] = ""
        sanitized_entry["meta_data"]["model_name"] = entry["meta_data"]["model_name"]
        sanitized_entry["meta_data"]["artifacts_file"] = ""
        sanitized_entry["meta_data"]["synthesis_info"] = [
            {
                "start_time": entry["meta_data"]["synthesis_start"],
                "end_time": entry["meta_data"]["synthesis_end"],
                "cpu": entry["meta_data"]["cpu"],
                "build_args": {
                    "reset": False,
                    "csim": False,
                    "synth": True,
                    "cosim": False,
                    "validation": False,
                    "export": False,
                    "vsynth": False,
                    "fifo_opt": False,
                    "bitfile": False,
                },
            }
        ]

        sanitized_entry["model_config"] = entry["model_config"]
        sanitized_entry["hls_config"] = entry["hls_config"]
        del sanitized_entry["hls_config"]["board"]

        sanitized_entry["resource_report"] = {}
        sanitized_entry["resource_report"]["CSynthesisReport"] = entry["resource_report"]

        sanitized_entry["latency_report"] = entry["latency_report"]

        sanitized_entry["target_part"] = entry["target_part"]
        sanitized_entry["backend"] = entry["backend"]
        sanitized_entry["backend_version"] = entry["backend_version"]
        sanitized_entry["hls4ml_version"] = entry["hls4ml_version"]

        sanitized_entry["meta_data"]["model_id"] = md5_hash_dict(
            {
                "model_config": sanitized_entry["model_config"],
                "hls_config": sanitized_entry["hls_config"],
            }
        )
        sanitized_entry["meta_data"][
            "artifacts_file"
        ] = f"{sanitized_entry['meta_data']['model_id']}.tar.gz"

        if save_files:
            old_model_id = entry["meta_data"]["model_id"]
            old_json_path = os.path.join(output_dir, f"{old_model_id}.json")
            if os.path.exists(old_json_path):
                os.remove(old_json_path)

            new_model_id = sanitized_entry["meta_data"]["model_id"]
            new_json_path = os.path.join(output_dir, f"{new_model_id}.json")
            save_to_json(sanitized_entry, new_json_path, indent=2)

            if os.path.isdir(os.path.join(output_dir, "projects", str(old_model_id))):
                os.rename(
                    os.path.join(output_dir, "projects", str(old_model_id)),
                    os.path.join(output_dir, "projects", str(new_model_id)),
                )

            if os.path.exists(os.path.join(output_dir, "projects", f"{old_model_id}.tar.gz")):
                os.rename(
                    os.path.join(output_dir, "projects", f"{old_model_id}.tar.gz"),
                    os.path.join(output_dir, "projects", f"{new_model_id}.tar.gz"),
                )

        sanitized_data.append(sanitized_entry)

        if progress_counter is not None and progress_lock is not None:
            with progress_lock:
                progress_counter.value += 1

    return sanitized_data


def change_json_file_keys(args):
    data = args["json_data"]
    output_dir = args["output_dir"]
    save_files = args.get("save_files", False)

    progress_counter = args.get("progress_counter", None)
    progress_lock = args.get("progress_lock", None)

    changed_data = []
    for entry in data:
        changed_entry = {}
        changed_entry["meta_data"] = entry["meta_data"]
        changed_entry["model_config"] = entry["model_config"]
        changed_entry["hls_config"] = entry["hls_config"]

        changed_entry["hls_resource_report"] = {}
        for key in entry["resource_report"]["CSynthesisReport"]:
            changed_entry["hls_resource_report"][key.lower()] = entry["resource_report"][
                "CSynthesisReport"
            ][key]

        changed_entry["resource_report"] = {}
        for key in entry["resource_report"]["VivadoSynthReport"]:
            changed_entry["resource_report"][key.lower()] = entry["resource_report"][
                "VivadoSynthReport"
            ][key]

        changed_entry["latency_report"] = entry["latency_report"]
        changed_entry["target_part"] = entry["target_part"]
        changed_entry["backend"] = entry["backend"]
        changed_entry["backend_version"] = entry["backend_version"]
        changed_entry["hls4ml_version"] = entry["hls4ml_version"]

        if save_files:
            output_json_path = os.path.join(
                output_dir, f"{changed_entry['meta_data']['model_id']}.json"
            )
            save_to_json(changed_entry, output_json_path, indent=2)

        changed_data.append(changed_entry)

        if progress_counter is not None and progress_lock is not None:
            with progress_lock:
                progress_counter.value += 1

    return changed_data


def read_report_from_tar(pattern, tar_path):
    report_content = ""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if fnmatch.fnmatch(member.name, pattern):
                    report_file = tar.extractfile(member)
                    if report_file:
                        report_content = report_file.read().decode("utf-8")
                        break
    except Exception as e:
        print(f"Error reading tar file {tar_path}: {e}")
        return ""

    return report_content


def extract_ii_from_csynth_report(args):
    json_data = args["json_data"]
    base_dir = args["base_dir"]
    output_dir = args["output_dir"]
    save_files = args.get("save_files", False)

    csynth_counter = args.get("csynth_counter", None)
    csynth_lock = args.get("csynth_lock", None)

    reports_counter = args.get("reports_counter", None)
    reports_lock = args.get("reports_lock", None)

    progress_counter = args.get("progress_counter", None)
    progress_lock = args.get("progress_lock", None)

    header_pattern = re.compile(
        r"(\s*\|\s*latency\s*[a-zA-Z0-9()]*)+\s*\|\s*interval\s*\|\s*pipeline\s*\|\s*"
    )
    # interval_pattern = re.compile(r"\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*\w*\s*\|\s*")
    interval_pattern = re.compile(r"([-+]?\d*\.\d+|[-+]?\d+)")

    project_dir = os.path.join(base_dir, "projects")
    for entry in json_data:
        model_id = entry["meta_data"].get("model_id", None)
        if model_id is None:
            model_id = entry["meta_data"]["artifacts_file"].split(".")[0]
        if "CSynthesisReport" in entry["resource_report"]:
            if csynth_counter is not None and csynth_lock is not None:
                with csynth_lock:
                    csynth_counter.value += 1

        project_file = os.path.join(project_dir, f"{model_id}.tar.gz")
        if not os.path.isfile(project_file):
            continue

        report_pattern = f"*{model_id}*/myproject_prj/solution1/syn/report/myproject_csynth.rpt"
        report_content = read_report_from_tar(report_pattern, project_file)
        if report_content:
            report_lines = report_content.split("\n")
            header_found = False
            for line in report_lines:
                line = line.lower().strip()
                if header_pattern.match(line):
                    header_found = True
                    continue

                if header_found:
                    numbers = interval_pattern.findall(line)
                    if numbers:
                        interval_min = float(numbers[-2])
                        interval_max = float(numbers[-1])
                        entry["latency_report"]["interval_min"] = str(interval_min)
                        entry["latency_report"]["interval_max"] = str(interval_max)
                        break

            if reports_counter is not None and reports_lock is not None:
                with reports_lock:
                    reports_counter.value += 1

        if save_files:
            # if "VivadoSynthReport" in entry["resource_report"]:
            output_json_path = os.path.join(output_dir, f"{model_id}.json")
            save_to_json(entry, output_json_path, indent=2)

        if progress_counter is not None and progress_lock is not None:
            with progress_lock:
                progress_counter.value += 1

    # print(f"Found {found_reports} reports, csynth jsons total: {jsons_with_csynth}, total jsons: {len(json_data)}")


if __name__ == "__main__":
    pass

    # split_data = read_from_json(
    #     os.path.join(base_path, "datasets", "iccad_submit", "preprocessed", "benchmark", "*.json"),
    # )
    # output_file = os.path.join(base_path, "datasets", "iccad_submit", "preprocessed", "benchmark_vsynth_with_ii.json")
    # with open(output_file, "w") as json_file:
    #     json.dump(split_data, json_file, indent=2)

    # extract_ii_from_csynth_report({
    #     "input_dir": os.path.join(base_path, "datasets", "vsynth_2"),
    #     "output_dir": os.path.join(base_path, "datasets", "vsynth_2"),
    #     "save_files": True,
    #     "progress_counter": None,
    #     "progress_lock": None,
    # })

    # gen_settings = GeneratorSettings(
    #     input_range=Power2Range(16, 32),
    #     layer_range=IntRange(2, 3),
    #     neuron_range=Power2Range(16, 32),
    #     output_range=IntRange(1, 20),
    #     activations=["relu"],
    # )
    # synth_settings = SynthSettings(
    #     reuse_range=Power2Range(32, 64),
    #     precisions=["ap_fixed<2, 1>", "ap_fixed<8, 3>"],
    #     strategies=["Resource"],
    # )

    # n_procs = 3
    # with Pool(n_procs) as p:
    #     result = p.map_async(
    #         parallel_rnd_synthesis,
    #         [
    #             {
    #                 "job_id": f"{proc}",
    #                 "n_models": 1,
    #                 "prj_compress": True,
    #                 "gen_settings": gen_settings,
    #                 "synth_settings": synth_settings,
    #                 "build_args": {
    #                     "reset": False,
    #                     "csim": False,
    #                     "cosim": False,
    #                     "synth": True,
    #                     "vsynth": False,
    #                     "validation": False,
    #                     "export": False,
    #                     "fifo_opt": False,
    #                     "bitfile": False
    #                 }
    #             }
    #             for proc in range(1, n_procs + 1)
    #         ],
    #     )
    #     while not result.ready():
    #         time.sleep(1)
    #     result = result.get()
    #     p.terminate()
    #     p.join()

    # add_data = read_from_json(json_path, ParsedDataFilter(include_layers=["Add"], exclude_layers=["Concatenate"]))
    # concat_data = read_from_json(json_path, ParsedDataFilter(include_layers=["Concatenate"], exclude_layers=["Add"]))
    # add_concat_data = read_from_json(json_path, ParsedDataFilter(include_layers=["Add", "Concatenate"]))
    # all_skip_data = add_data + concat_data + add_concat_data
    # file_path = os.path.join(base_path, "datasets", "fcnn_skip_dataset.json")
    # with open(file_path, "w") as json_file:
    #     json.dump(all_skip_data, json_file, indent=2)
    # json_data = read_from_json(json_path, ParsedDataFilter(exclude_layers=["Add", "Concatenate"]))
    # split_data = np.array_split(json_data, 4)
    # for i, data in enumerate(split_data):
    #     file_path = os.path.join(base_path, "datasets", f"fcnn_dataset-{i+1}.json")
    #     with open(file_path, "w") as json_file:
    #         json.dump(data.tolist(), json_file, indent=2)

    # json_path = os.path.join(base_path, "datasets", "vsynth_2", "*.json")
    # json_path = os.path.join(base_path, "datasets", "vsynth_3_4", "*.json")
    # json_path = os.path.join(base_path, "datasets", "*.json")
    # json_path = os.path.join(base_path, "datasets", "iccad_submit", "*.json")
    # json_path = os.path.join(base_path, "datasets", "iccad_submit", "vsynth_2", "*.json")
    # json_path = os.path.join(base_path, "datasets", "iccad_submit", "vsynth_3_4", "*.json")
    # json_path = os.path.join(base_path, "datasets", "benchmark", "*.json")
    # json_data = read_from_json(json_path)

    # def archive_vsynth_jsons(json_path):
    #     json_data = read_from_json(json_path)
    #     csynth_count = 0
    #     vsynth_count = 0
    #     vsynth_json_paths = []
    #     for model_entry in json_data:
    #         res_report = model_entry["resource_report"]
    #         if "CSynthesisReport" in res_report:
    #             csynth_count += 1
    #         if "VivadoSynthReport" in res_report:
    #             vsynth_count += 1
    #             json_path = os.path.join(os.path.dirname(json_path), f"{model_entry['meta_data']['model_id']}.json")
    #             if os.path.isfile(json_path):
    #                 vsynth_json_paths.append(json_path)

    #     json_dir = os.path.dirname(json_path)
    #     all_json_archive = os.path.join(json_dir, os.path.basename(json_dir) + "_jsons.tar.gz")
    #     with tarfile.open(all_json_archive, "w:gz") as tar:
    #         for json_path in tqdm(vsynth_json_paths):
    #             tar.add(json_path, arcname=os.path.basename(json_path))

    #     print(f"Total: {len(json_data)}, CSynth: {csynth_count}, VSynth: {vsynth_count}")

    # def archive_vsynth_projects(json_path):
    #     json_data = read_from_json(json_path)
    #     csynth_count = 0
    #     vsynth_count = 0
    #     project_dir = os.path.join(os.path.dirname(json_path), "projects")
    #     vsynth_tar_paths = []
    #     for model_entry in json_data:
    #         res_report = model_entry["resource_report"]
    #         if "CSynthesisReport" in res_report:
    #             csynth_count += 1
    #         if "VivadoSynthReport" in res_report:
    #             vsynth_count += 1
    #             tar_path = os.path.join(project_dir, f"{model_entry['meta_data']['model_id']}.tar.gz")
    #             if os.path.isfile(tar_path):
    #                 vsynth_tar_paths.append(tar_path)

    #     all_tars_archive = os.path.join(project_dir, os.path.basename(os.path.dirname(json_path)) + "_projects.tar.gz")
    #     with tarfile.open(all_tars_archive, "w:gz") as tar:
    #         for tar_path in tqdm(vsynth_tar_paths):
    #             tar.add(tar_path, arcname=os.path.basename(tar_path))

    #     print(f"Total: {len(json_data)}, CSynth: {csynth_count}, VSynth: {vsynth_count}")

    # archive_vsynth_jsons(os.path.join(base_path, "datasets", "vsynth_2", "*.json"))
    # archive_vsynth_jsons(os.path.join(base_path, "datasets", "vsynth_3_4", "*.json"))

    # archive_vsynth_projects(os.path.join(base_path, "datasets", "vsynth_2", "*.json"))
    # archive_vsynth_projects(os.path.join(base_path, "datasets", "vsynth_3_4", "*.json"))

    # n_workers = min(16, len(json_data))

    # build_args = {
    #     "reset": False,
    #     "csim": False,
    #     "cosim": False,
    #     "synth": False,
    #     "vsynth": True,
    #     "validation": False,
    #     "export": False,
    #     "fifo_opt": False,
    #     "bitfile": False,
    # }

    # skip_ids = [
    #     # Path(f).stem for f in glob.glob(
    #     #     os.path.join(base_path, "datasets", "vsynth", "*.json")
    #     # )
    # ]

    # data_left_to_process = []
    # for model_entry in json_data:
    #     model_id = md5_hash_dict(
    #         {"model_config": model_entry["model_config"], "hls_config": model_entry["hls_config"]}
    #     )

    #     res_report = model_entry["resource_report"]
    #     target_reports = []
    #     if build_args["synth"]:
    #         target_reports.append("CSynthesisReport")
    #     if build_args["vsynth"]:
    #         target_reports.append("VivadoSynthReport")

    #     if model_id not in skip_ids and not all(report in res_report for report in target_reports):
    #         data_left_to_process.append(model_entry)

    # data_splits = np.array_split(data_left_to_process, n_workers)

    # data_left_to_process = json_data
    # data_splits = np.array_split(data_left_to_process, n_workers)

    # with Manager() as manager:
    #     progress_counter = manager.Value("i", 0)
    #     progress_lock = manager.Lock()

    # csynth_counter = manager.Value("j", 0)
    # csynth_lock = manager.Lock()

    # reports_counter = manager.Value("k", 0)
    # reports_lock = manager.Lock()

    # with tqdm(total=len(data_left_to_process), desc="Synthesis Loop") as pbar:
    #     with Pool(n_workers) as p:
    # result = p.map_async(
    #     synth_from_json,
    #     [
    #         {
    #             "output_dir": os.path.join(base_path, "datasets", "vsynth_3_4"),
    #             "json_data": data_split,
    #             "prj_compress": True,
    #             "prj_decompress": True,
    #             "xilinx_path": "/opt/Xilinx",
    #             "build_args": build_args,
    #             "progress_counter": progress_counter,
    #             "progress_lock": progress_lock,
    #         }
    #         for idx, data_split in enumerate(data_splits)
    #     ],
    # )
    # result = p.map_async(
    #     sanitize_csynth_data,
    #     [
    #         {
    #             "json_data": data_split,
    #             "output_dir": os.path.join(base_path, "datasets", "vsynth_3_4"),
    #             "save_files": True,
    #             "progress_counter": progress_counter,
    #             "progress_lock": progress_lock,
    #         }
    #         for idx, data_split in enumerate(data_splits)
    #     ],
    # )
    # result = p.map_async(
    #     change_json_file_keys,
    #     [
    #         {
    #             "json_data": data_split,
    #             "output_dir": os.path.join(base_path, "datasets", "iccad_submit", "preprocessed"),
    #             "save_files": True,
    #             "progress_counter": progress_counter,
    #             "progress_lock": progress_lock,
    #         }
    #         for idx, data_split in enumerate(data_splits)
    #     ],
    # )
    # result = p.map_async(
    #     extract_ii_from_csynth_report,
    #     [
    #         {
    #             "json_data": data_split,
    #             "base_dir": os.path.join(base_path, "datasets", "benchmark"),
    #             "output_dir": os.path.join(base_path, "datasets", "iccad_submit", "preprocessed", "benchmark"),
    #             "save_files": True,
    #             # "csynth_counter": csynth_counter,
    #             # "csynth_lock": csynth_lock,
    #             # "reports_counter": reports_counter,
    #             # "reports_lock": reports_lock,
    #             "progress_counter": progress_counter,
    #             "progress_lock": progress_lock,
    #         }
    #         for idx, data_split in enumerate(data_splits)
    #     ],
    # )

    # while not result.ready():
    #     pbar.n = progress_counter.value
    #     pbar.refresh()
    # print(f"CSynth: {csynth_counter.value}, Reports: {reports_counter.value}")
    # time.sleep(1)

    # result = result.get()
    # pbar.n = progress_counter.value
    # pbar.refresh()

    # print(f"Total: {len(data_left_to_process)}, CSynth: {csynth_counter.value}, Reports: {reports_counter.value}")

    # p.terminate()
    # p.join()

    # data_split = data_splits[0]
    # synth_from_json(
    #     {
    #         "output_dir": os.path.join(base_path, "datasets", "vsynth"),
    #         "json_data": data_split,
    #         "prj_compress": True,
    #         "prj_decompress": True,
    #         "xilinx_path": "/opt/Xilinx",
    #         "build_args": build_args,
    #         # "progress_counter": progress_counter,
    #         # "progress_lock": progress_lock,
    #     }
    # )

    #     hls_config = model_entry["hls_config"]
    #     clock_period = str(int(float(model_entry["latency_report"]["target_clock"])))
    #     part_number = get_part_from_board(hls_config["board"])
    #     io_type = hls_config.get("io_type", "io_parallel")
    #     backend = model_entry.get("backend", "VivadoAccelerator")
    #     backend_version = model_entry.get("backend_version", "2019.1")

    #     synth_result = synthesize_keras_model(
    #         model,
    #         os.path.join(base_path, "datasets", "vsynth"),
    #         hls_config,
    #         clock_period,
    #         part_number,
    #         io_type,
    #         backend,
    #         backend_version,
    #         xilinx_path="/opt/Xilinx",
    #         verbose=1,
    #         reset=False,
    #         csim=False,
    #         cosim=False,
    #         synth=True,
    #         vsynth=False,
    #         validation=False,
    #         export=False,
    #         fifo_opt=False,
    #         bitfile=False
    #     )

    #     print(synth_result)
