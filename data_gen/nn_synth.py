import inspect
import json
import os
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime

import hls4ml
import numpy as np
import tensorflow as tf
from hls4ml.backends.backend import get_available_backends, get_backend
from nn_gen import GeneratorSettings, IntRange, Power2Range, generate_fc_network
from utils import (
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
        for arg in list(
            inspect.signature(get_backend(backend_name).build).parameters.values()
        )[1:]
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

    reuse_range: Power2Range = field(default_factory=lambda: Power2Range(1, 64))
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

    clock_period_range: IntRange = field(default_factory=lambda: IntRange(10, 10))
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
        raise ValueError(
            f"Backend {backend} not available. Choose from {available_backends}"
        )

    backend_frmt = backend[0].upper() + backend[1:].lower()
    if backend_frmt.endswith("accelerator"):
        backend_frmt = backend_frmt.replace("accelerator", "")

    backend_path = os.path.join(xilinx_path, backend_frmt, backend_version, "bin")
    if not os.path.exists(backend_path):
        raise FileNotFoundError(
            f'{backend_frmt} {backend_version} installation not found at "{backend_path}". '
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
    """

    add_backend_to_env(backend, backend_version, xilinx_path)

    model_config = config_from_keras_model(model, hls_config)

    updated_model_id = md5_hash_dict(
        {"model_config": model_config, "hls_config": hls_config}
    )
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
    """

    add_backend_to_env(backend, backend_version, xilinx_path)

    model_config = config_from_torch_model(model, hls_config)
    input_shape = model_config[0]["input_shape"]

    updated_model_id = md5_hash_dict(
        {"model_config": model_config, "hls_config": hls_config}
    )
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
    """Synthesize each model with each HLS config and save the reports."""

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
            backend_versions += [backend_versions[-1]] * (
                len(backends) - len(backend_versions)
            )

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
                                    meta_entry["synthesis_info"] += synth_result[
                                        "meta_data"
                                    ]["synthesis_info"]
                                    synth_result["meta_data"]["synthesis_info"] = (
                                        meta_entry["synthesis_info"]
                                    )

                                updated_model_id = synth_result["meta_data"]["model_id"]
                                if model_id is None:
                                    model_id = updated_model_id

                                json_file = (
                                    json_filename
                                    if json_filename
                                    else f"{model_id}.json"
                                )
                                json_path = os.path.join(output_dir, json_file)

                                if os.path.isfile(json_path):
                                    os.remove(json_path)
                                save_to_json(
                                    synth_result,
                                    os.path.join(
                                        output_dir, f"{updated_model_id}.json"
                                    ),
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
                                        os.path.join(
                                            project_dir, f"{updated_model_id}.tar.gz"
                                        ),
                                        "w:gz",
                                    ) as tar:
                                        tar.add(
                                            os.path.join(
                                                project_dir, str(updated_model_id)
                                            ),
                                            arcname=str(updated_model_id),
                                        )

                                    shutil.rmtree(
                                        os.path.join(project_dir, str(updated_model_id))
                                    )

                            except Exception as e:  # noqa: BLE001, keep the batch running
                                print(e)


def parallel_rnd_synthesis(args):
    """Generate random networks and synthesize them, one job of a parallel run."""

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
            hls_config = hls4ml.utils.config_from_keras_model(
                gen_network, granularity=granularity
            )

            model_precision = rng.choice(synth_settings.precisions)
            model_reuse_factor = int(synth_settings.reuse_range.random_in_range(rng))
            model_strategy = rng.choice(synth_settings.strategies)

            hls_config["Model"]["Precision"] = model_precision
            hls_config["Model"]["ReuseFactor"] = model_reuse_factor
            hls_config["Model"]["Strategy"] = model_strategy

            if granularity == "name":
                for layer_name in hls_config["LayerName"]:
                    layer_dict = {}
                    if isinstance(
                        hls_config["LayerName"][layer_name]["Precision"], dict
                    ):
                        for prec_key in hls_config["LayerName"][layer_name][
                            "Precision"
                        ]:
                            layer_dict["Precision"][prec_key] = rng.choice(
                                synth_settings.precisions
                            )
                    else:
                        layer_dict["Precision"] = rng.choice(synth_settings.precisions)

                    layer_dict["ReuseFactor"] = int(
                        synth_settings.reuse_range.random_in_range(rng)
                    )
                    layer_dict["Strategy"] = rng.choice(synth_settings.strategies)

                    hls_config["LayerName"][layer_name] = layer_dict
            elif granularity == "type":
                layer_type_dict = {}
                for layer_type in hls_config["LayerType"]:
                    if isinstance(
                        hls_config["LayerType"][layer_type]["Precision"], dict
                    ):
                        for prec_key in hls_config["LayerType"][layer_type][
                            "Precision"
                        ]:
                            layer_type_dict["Precision"][prec_key] = rng.choice(
                                synth_settings.precisions
                            )
                    else:
                        layer_type_dict["Precision"] = rng.choice(
                            synth_settings.precisions
                        )

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
        except Exception as e:  # noqa: BLE001, keep the batch running
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
            tar_path = os.path.join(
                args["output_dir"], "projects", f"{model_id}.tar.gz"
            )

            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(project_dir)

                    top_level_dirs = {
                        member.name.split("/")[0]
                        for member in tar.getmembers()
                        if member.name.count("/") > 0
                    }
                    if len(top_level_dirs) == 1:
                        extracted_dir = os.path.join(
                            project_dir, next(iter(top_level_dirs))
                        )
                    else:
                        raise ValueError(
                            f"Unexpected directory structure: {top_level_dirs}"
                        )

                    # rename the extracted project folder to the model_id
                    new_dir = os.path.join(project_dir, str(model_id))
                    if extracted_dir != new_dir:
                        os.rename(extracted_dir, new_dir)
            except (OSError, EOFError, ValueError, tarfile.TarError) as e:
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
