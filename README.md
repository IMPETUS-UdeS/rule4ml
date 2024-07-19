[![License](https://img.shields.io/badge/License-GPL_3.0-red.svg)](https://opensource.org/license/gpl-3-0)

# rule4ml: Resource Utilization and Latency Estimation for ML

`rule4ml` is a tool designed for pre-synthesis estimation of FPGA resource utilization and inference latency for machine learning models.

## Installation

`rule4ml` can be installed using the provided `setup.py`. A package will soon be available on the Python Package Index for easier installation via `pip` directly.

```bash
git clone git@github.com:IMPETUS-UdeS/rule4ml.git

cd rule4ml

pip install ./
```

This will only install the base package and its dependencies. The dependencies for the [data_gen](data_gen/) scripts, used for generating synthetic data, need to be installed separately. These dependencies are listed in [requirements.txt](data_gen/requirements.txt).

## Getting Started

### Tutorial
To get started with `rule4ml`, refer to the detailed Jupyter Notebook [tutorial](notebooks/tutorial.ipynb). This tutorial covers:

- Using pre-trained estimators for resources and latency predictions.
- Generating synthetic datasets.
- Training your own predictors.

### Usage
Here's a quick example of how to use `rule4ml` to estimate resources and latency for a given model:

```python
import keras
from keras.layers import Input, Dense, Activation

from rule4ml.models.estimators import MultiModelEstimator

# Example of a simple keras Model
input_size = 16
inputs = Input(shape=(input_size,))
x = Dense(32, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(5, activation="softmax")(x)

model_to_predict = keras.Model(inputs=inputs, outputs=outputs, name="Jet Classifier")
model_to_predict.build((None, input_size))

# Loading default predictors
estimator = MultiModelEstimator()
estimator.load_default_models()

# Predictions are formatted as a pandas DataFrame
prediction_df = estimator.predict(model_to_predict)

# Outside of Jupyter notebooks, we recommend saving the DataFrame as HTML for better readability
prediction_df.to_html("keras_example.html")
```

## Datasets
Training accurate predictors requires large datasets of synthesized neural networks. We used [hls4ml](https://github.com/fastmachinelearning/hls4ml) to synthesize neural networks generated with parameters randomly sampled from predefined ranges. The data our models were trained on can be found at [https://borealisdata.ca/dataverse/rule4ml](https://borealisdata.ca/dataverse/rule4ml).

## Limitations
In their current iteration, the predictors can process [Keras](https://keras.io/about/) or [PyTorch](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) models to generate FPGA resources (**BRAM**, **DSP**, **FF**, **LUT**) and latency (**Clock Cycles**) estimations for various synthesis configurations. However, the training models are limited to specific layers: **Dense/Linear**, **ReLU**, **Tanh**, **Sigmoid**, **Softmax**, **BatchNorm**, **Add**, **Concatenate**, and **Dropout**. They are also constrained by synthesis parameters, notably **clock_period** (10 ns) and **io_type** (io_parallel). Inputs outside these configurations may result in inaccurate predictions.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
