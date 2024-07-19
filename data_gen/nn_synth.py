import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import hls4ml
import numpy as np
import tensorflow as tf
from nn_gen import GeneratorSettings, generate_fc_network
from utils import (
    IntRange,
    Power2Range,
    data_from_synthesis,
    get_cpu_info,
    print_hls_config,
    save_to_json,
)

from rule4ml.parsers.network_parser import config_from_keras_model, config_from_torch_model
from rule4ml.parsers.utils import camel_keys_to_snake

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Specify Vivado path (alternatively source settings64.sh)
os.environ["PATH"] = "/opt/Xilinx/Vivado/2019.1/bin:" + os.environ["PATH"]


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

    # Experiment with clock_period and io_type in the future
    clock_period_range: IntRange = IntRange(10, 10)
    io_types: list = field(default_factory=lambda: ["io_parallel"])


def keras_to_hls(
    model,
    output_dir,
    board,
    strategy,
    precision,
    reuse_factor,
    clock_period,
    io_type,
    verbose=0,
):
    """
    Creates an hls4ml Model using a keras Model and hls4ml settings.

    Args:
        model (keras.Model): _description_
        output_dir (str): _description_
        precision (str): _description_
        strategy (str): _description_
        reuse_factor (int): _description_
        board (str): _description_
        clock_period (str): _description_
        io_type (str): _description_
        verbose (int, optional): Output verbosity. Defaults to 0.

    Returns:
        hls4ml.Model: _description_
        dict: _description_
    """

    config = hls4ml.utils.config_from_keras_model(model, granularity="model")

    config["Model"]["Precision"] = precision
    config["Model"]["Strategy"] = strategy
    config["Model"]["ReuseFactor"] = reuse_factor

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend="VivadoAccelerator",
        clock_period=clock_period,
        io_type=io_type,
        board=board,
    )
    hls_model.compile()

    config = camel_keys_to_snake(config, recursive=True)

    config["clock_period"] = float(clock_period)
    config["io_type"] = io_type
    config["board"] = board

    if verbose > 0:
        print("-----------------------------------")
        print("Configuration")
        print_hls_config(config)
        print("-----------------------------------")

    return hls_model, config


def torch_to_hls(
    model,
    input_shape,
    output_dir,
    board,
    strategy,
    precision,
    reuse_factor,
    clock_period,
    io_type,
    verbose=0,
):
    """
    Creates an hls4ml Model using a torch Model and hls4ml settings.

    Args:
        model (torch.nn.Module): _description_
        input_shape (list|tuple): _description_
        output_dir (str): _description_
        precision (str): _description_
        strategy (str): _description_
        reuse_factor (int): _description_
        board (str): _description_
        clock_period (str): _description_
        io_type (str): _description_
        verbose (int, optional): Output verbosity. Defaults to 0.

    Returns:
        hls4ml.Model: _description_
        dict: _description_
    """

    config = hls4ml.utils.config_from_pytorch_model(model, granularity="model")

    config["Model"]["Precision"] = precision
    config["Model"]["Strategy"] = strategy
    config["Model"]["ReuseFactor"] = reuse_factor

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape,
        hls_config=config,
        output_dir=output_dir,
        backend="VivadoAccelerator",
        clock_period=clock_period,
        io_type=io_type,
        board=board,
    )
    hls_model.compile()

    config = camel_keys_to_snake(config, recursive=True)

    config["clock_period"] = float(clock_period)
    config["io_type"] = io_type
    config["board"] = board

    if verbose > 0:
        print("-----------------------------------")
        print("Configuration")
        print_hls_config(config)
        print("-----------------------------------")

    return hls_model, config


def synthesize_keras_model(
    model,
    board,
    strategy,
    precision,
    reuse_factor,
    clock_period,
    io_type,
    project_dir="./hls4ml_prj",
    synth_uuid=None,
    verbose=0,
):
    """
    _summary_

    Args:
        model (_type_): _description_
        board (_type_): _description_
        strategy (_type_): _description_
        precision (_type_): _description_
        reuse_factor (_type_): _description_
        clock_period (_type_): _description_
        io_type (_type_): _description_
        project_dir (str, optional): _description_. Defaults to "./hls4ml_prj".
        synth_uuid (_type_, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    model_config = config_from_keras_model(model, reuse_factor)

    meta_data = {}
    if synth_uuid is not None:
        project_dir += f"_{synth_uuid}"
        meta_data["uuid"] = str(synth_uuid)

    hls_model, hls_config = keras_to_hls(
        model,
        project_dir,
        board,
        strategy,
        precision,
        reuse_factor,
        clock_period,
        io_type,
        verbose=verbose,
    )

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    synth_result = hls_model.build(csim=False, synth=True, export=False, bitfile=False)
    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    resource_report, latency_report = data_from_synthesis(synth_result)

    # Alternatively use regex to get info from report file
    # resource_report, latency_report = data_from_report(os.path.join(
    #     project_dir,
    #     "myproject_prj/solution1/syn/report/myproject_axi_csynth.rpt"
    # ))

    del model
    del hls_model
    tf.keras.backend.clear_session()

    if synth_uuid is None:
        meta_data["uuid"] = str(uuid.uuid4())
    meta_data["synthesis_start"] = start_time
    meta_data["synthesis_end"] = end_time
    meta_data["cpu"] = get_cpu_info()

    return {
        "meta_data": meta_data,
        "model_config": model_config,
        "hls_config": hls_config,
        "resource_report": resource_report,
        "latency_report": latency_report,
    }


def synthesize_torch_model(
    model,
    board,
    strategy,
    precision,
    reuse_factor,
    clock_period,
    io_type,
    project_dir="./hls4ml_prj",
    synth_uuid=None,
    verbose=0,
):
    """
    _summary_

    Args:
        model (_type_): _description_
        board (_type_): _description_
        strategy (_type_): _description_
        precision (_type_): _description_
        reuse_factor (_type_): _description_
        clock_period (_type_): _description_
        io_type (_type_): _description_
        project_dir (str, optional): _description_. Defaults to "./hls4ml_prj".
        synth_uuid (_type_, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.
    """

    model_config = config_from_torch_model(model, reuse_factor)
    input_shape = model_config[0]["input_shape"]

    meta_data = {}
    if synth_uuid is not None:
        project_dir += f"_{synth_uuid}"
        meta_data["uuid"] = str(synth_uuid)

    hls_model, hls_config = torch_to_hls(
        model,
        input_shape,
        project_dir,
        board,
        strategy,
        precision,
        reuse_factor,
        clock_period,
        io_type,
        verbose=verbose,
    )

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    synth_result = hls_model.build(csim=False, synth=True, export=False, bitfile=False)
    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    resource_report, latency_report = data_from_synthesis(synth_result)

    del model
    del hls_model

    if synth_uuid is None:
        meta_data["uuid"] = str(uuid.uuid4())
    meta_data["synthesis_start"] = start_time
    meta_data["synthesis_end"] = end_time
    meta_data["cpu"] = get_cpu_info()

    return {
        "meta_data": meta_data,
        "model_config": model_config,
        "hls_config": hls_config,
        "resource_report": resource_report,
        "latency_report": latency_report,
    }


def parallel_generative_synthesis(args):
    """
    _summary_

    Args:
        args (_type_): _description_
    """

    job_id = args["job_id"]
    n_models = args.get("n_models", 0)

    prj_overwrite = args.get("prj_overwrite", False)
    project_dir = args.get("project_dir", os.path.join(base_path, "datasets/projects"))
    project_dir = os.path.join(project_dir, "hls4ml_prj")
    project_dir += f"_{job_id}"

    save_path = args.get("save_path", os.path.join(base_path, "datasets"))
    save_path = os.path.join(save_path, f"dataset-{job_id}.json")

    rng_seed = args.get("rng_seed", None)
    rng = np.random.default_rng(rng_seed)

    gen_function = args.get("gen_function", generate_fc_network)
    gen_settings = args.get("gen_settings", GeneratorSettings())
    gen_verbose = args.get("gen_verbose", 0)

    synth_function = args.get("synth_function", synthesize_keras_model)
    synth_settings = args.get("synth_settings", SynthSettings())
    synth_verbose = args.get("synth_verbose", 0)

    idx = 0
    synth_uuid = None
    while True:
        try:
            gen_network = gen_function(gen_settings, rng=rng, verbose=gen_verbose)

            if not prj_overwrite:
                synth_uuid = uuid.uuid4()

            board = rng.choice(synth_settings.boards)
            strategy = rng.choice(synth_settings.strategies)
            reuse_factor = int(synth_settings.reuse_range.random_in_range(rng))
            precision = rng.choice(synth_settings.precisions)

            clock_period = str(synth_settings.clock_period_range.random_in_range(rng))
            io_type = rng.choice(synth_settings.io_types)

            data_dict = synth_function(
                gen_network,
                board=board,
                strategy=strategy,
                precision=precision,
                reuse_factor=reuse_factor,
                clock_period=clock_period,
                io_type=io_type,
                project_dir=project_dir,
                synth_uuid=synth_uuid,
                verbose=synth_verbose,
            )
            data_dict["meta_data"]["job_id"] = job_id
            save_to_json(data_dict, save_path, indent=2)

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
    # pool = ProcessPoolExecutor()
    # pool.map(
    #     parallel_generative_synthesis,
    #     [
    #         {
    #             "job_id": f"{proc}",
    #             "n_models": 100,
    #             "prj_overwrite": False,
    #             "gen_settings": gen_settings,
    #             "synth_settings": synth_settings,
    #         }
    #         for proc in range(1, n_procs + 1)
    #     ],
    # )
    # pool.shutdown()

    # n_procs = 3
    # with Pool(n_procs) as p:
    #     result = p.map_async(
    #         parallel_generative_synthesis,
    #         [
    #             {
    #                 "job_id": f"{proc}",
    #                 "n_models": 100,
    #                 "prj_overwrite": False,
    #                 # "gen_settings": gen_settings,
    #                 "synth_settings": synth_settings,
    #             }
    #             for proc in range(1, n_procs + 1)
    #         ],
    #     )
    #     while not result.ready():
    #         time.sleep(1)
    #     result = result.get()
    #     p.terminate()
    #     p.join()
