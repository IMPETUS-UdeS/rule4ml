import inspect
import os
import shutil
import tarfile
import uuid
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
    model_name_from_config,
    print_hls_config,
    save_to_json,
)

from rule4ml.parsers.network_parser import config_from_keras_model, config_from_torch_model
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
    model_uuid=None,
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
        model_uuid (_type_, optional): _description_. Defaults to None.
        model_name (str, optional): _description_. Defaults to "".
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        dict: _description_
    """

    add_backend_to_env(backend, backend_version, xilinx_path)

    model_config = config_from_keras_model(model, hls_config)

    if model_uuid is None:
        model_uuid = uuid.uuid4()

    if not model_name:
        model_name = model_name_from_config(model, model_config, hls_config)

    meta_data = {
        "uuid": str(model_uuid),
        "model_name": model_name,
        "artifacts_file": f"{model_uuid}.tar.gz",
    }

    project_dir = os.path.join(output_dir, "projects", str(model_uuid))
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

    meta_data.update(
        {
            "synthesis_start": start_time,
            "synthesis_end": end_time,
            "cpu": get_cpu_info(),
        }
    )

    vsynth = build_kwargs.get("vsynth", False)
    resource_report, latency_report = data_from_synthesis(synth_result, vsynth)

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
    model_uuid=None,
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
        model_uuid (_type_, optional): _description_. Defaults to None.
        model_name (str, optional): _description_. Defaults to "".
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        dict: _description_
    """

    add_backend_to_env(backend, backend_version, xilinx_path)

    model_config = config_from_torch_model(model, hls_config)
    input_shape = model_config[0]["input_shape"]

    if model_uuid is None:
        model_uuid = uuid.uuid4()

    if not model_name:
        model_name = model_name_from_config(model, model_config, hls_config)

    meta_data = {
        "uuid": str(model_uuid),
        "model_name": model_name,
        "artifacts_file": f"{model_uuid}.tar.gz",
    }

    project_dir = os.path.join(output_dir, "projects", str(model_uuid))
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

    meta_data.update(
        {
            "synthesis_start": start_time,
            "synthesis_end": end_time,
            "cpu": get_cpu_info(),
        }
    )

    vsynth = build_kwargs.get("vsynth", False)
    resource_report, latency_report = data_from_synthesis(synth_result, vsynth)

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

    models = args["models"]
    hls_configs = args["hls_configs"]
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

    for model in models:
        for hls_config in hls_configs:
            for part_number in parts:
                for clock_period in clock_periods:
                    for io_type in io_types:
                        for backend, backend_version in zip(backends, backend_versions):
                            try:
                                model_uuid = uuid.uuid4()
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
                                    model_uuid=model_uuid,
                                    verbose=synth_verbose,
                                    **build_args,
                                )

                                json_file = json_filename if json_filename else f"{model_uuid}.json"
                                save_to_json(
                                    synth_result, os.path.join(output_dir, json_file), indent=2
                                )

                                if prj_compress:
                                    project_dir = os.path.join(output_dir, "projects")
                                    os.remove(os.path.join(project_dir, f"{model_uuid}.tar.gz"))

                                    with tarfile.open(
                                        os.path.join(project_dir, f"{model_uuid}.tar.gz"), "w:gz"
                                    ) as tar:
                                        tar.add(
                                            os.path.join(project_dir, str(model_uuid)),
                                            arcname=str(model_uuid),
                                        )

                                    shutil.rmtree(os.path.join(project_dir, str(model_uuid)))

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


if __name__ == "__main__":
    gen_settings = GeneratorSettings(
        input_range=Power2Range(16, 32),
        layer_range=IntRange(2, 3),
        neuron_range=Power2Range(16, 32),
        output_range=IntRange(1, 20),
        activations=["relu"],
    )
    synth_settings = SynthSettings(
        reuse_range=Power2Range(32, 64),
        precisions=["ap_fixed<2, 1>", "ap_fixed<8, 3>"],
        strategies=["Resource"],
    )

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

    from rule4ml.parsers.data_parser import read_from_json

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

    json_path = os.path.join(base_path, "datasets", "fcnn_dataset_15000.json")
    json_data = read_from_json(
        [
            os.path.join(base_path, "datasets", "fcnn_dataset-1.json"),
            os.path.join(base_path, "datasets", "fcnn_dataset-2.json"),
            os.path.join(base_path, "datasets", "fcnn_dataset-3.json"),
            os.path.join(base_path, "datasets", "fcnn_dataset-4.json"),
            os.path.join(base_path, "datasets", "fcnn_skip_dataset.json"),
        ]
    )
    print(len(json_data))

    # for model_entry in json_data[:1]:
    #     model_config = model_entry["model_config"]
    #     hls_config = model_entry["hls_config"]
    #     clock_period = str(int(model_entry["latency_report"]["target_clock"]))
    #     part_number = get_part_from_board(hls_config["board"])
    #     io_type = hls_config.get("io_type", "io_parallel")
    #     backend = model_entry.get("backend", "VivadoAccelerator")
    #     backend_version = model_entry.get("backend_version", "2019.1")

    #     model = keras_model_from_config(model_config)
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
