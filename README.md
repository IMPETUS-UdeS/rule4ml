[![License](https://img.shields.io/badge/License-GPL_3.0-red.svg)](https://opensource.org/license/gpl-3-0)

# rule4ml: Resource Utilization and Latency Estimation for ML

`rule4ml` is a tool designed for pre-synthesis estimation of FPGA resource utilization and inference latency for machine learning models.

## Installation

`rule4ml` releases are uploaded to the Python Package Index for easy installation via `pip`.

```bash
pip install rule4ml
```

This will only install the [base package](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/rule4ml) and its dependencies for resources and latency prediction. The [data_gen](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/data_gen/) scripts and the [Jupyter notebooks](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/notebooks) are to be cloned from the repo if needed. The data generation dependencies are listed seperately in [data_gen/requirements.txt](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/data_gen/requirements.txt).

## Getting Started

### Tutorial
To get started with `rule4ml`, please refer to the detailed Jupyter Notebook [tutorial](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/notebooks/tutorial.ipynb). This tutorial covers:

- Using pre-trained estimators for resources and latency predictions.
- Generating synthetic datasets.
- Training and testing your own predictors.

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
model_to_predict.build((None, input_size))  # building keras models is required

# Loading default predictors
estimator = MultiModelEstimator()
estimator.load_default_models()

# MultiModelEstimator predictions are formatted as a pandas DataFrame
prediction_df = estimator.predict(model_to_predict)

# Further formatting can applied to organize the DataFrame
if not prediction_df.empty:
    prediction_df = prediction_df.groupby(
        ["Model", "Board", "Strategy", "Precision", "Reuse Factor"], observed=True
    ).mean()  # each row is unique in the groupby, mean() is only called to convert DataFrameGroupBy

# Outside of Jupyter notebooks, we recommend saving the DataFrame as HTML for better readability
prediction_df.to_html("keras_example.html")
```

**keras_example.html** (truncated)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>BRAM (%)</th>
      <th>DSP (%)</th>
      <th>FF (%)</th>
      <th>LUT (%)</th>
      <th>CYCLES</th>
    </tr>
    <tr>
      <th>Model</th>
      <th>Board</th>
      <th>Strategy</th>
      <th>Precision</th>
      <th>Reuse Factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="21" valign="top">Jet Classifier</th>
      <th rowspan="21" valign="top">pynq-z2</th>
      <th rowspan="21" valign="top">Latency</th>
      <th rowspan="7" valign="top">ap_fixed&lt;2, 1&gt;</th>
      <th>1</th>
      <td>2.77</td>
      <td>0.89</td>
      <td>2.63</td>
      <td>30.02</td>
      <td>54.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.75</td>
      <td>0.86</td>
      <td>2.62</td>
      <td>29.91</td>
      <td>55.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.70</td>
      <td>0.79</td>
      <td>2.58</td>
      <td>29.80</td>
      <td>55.78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.97</td>
      <td>0.67</td>
      <td>2.49</td>
      <td>29.79</td>
      <td>68.84</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.97</td>
      <td>0.63</td>
      <td>2.50</td>
      <td>30.24</td>
      <td>75.38</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2.26</td>
      <td>0.74</td>
      <td>2.43</td>
      <td>30.90</td>
      <td>76.19</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.83</td>
      <td>0.47</td>
      <td>2.19</td>
      <td>32.89</td>
      <td>112.04</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">ap_fixed&lt;8, 3&gt;</th>
      <th>1</th>
      <td>2.63</td>
      <td>1.58</td>
      <td>13.91</td>
      <td>115.89</td>
      <td>53.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.63</td>
      <td>1.50</td>
      <td>13.63</td>
      <td>111.75</td>
      <td>54.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.59</td>
      <td>1.25</td>
      <td>13.07</td>
      <td>108.52</td>
      <td>56.16</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.76</td>
      <td>1.41</td>
      <td>12.22</td>
      <td>108.01</td>
      <td>53.07</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3.42</td>
      <td>1.96</td>
      <td>11.98</td>
      <td>104.58</td>
      <td>64.71</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2.99</td>
      <td>1.93</td>
      <td>12.74</td>
      <td>94.71</td>
      <td>83.06</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.56</td>
      <td>1.70</td>
      <td>14.74</td>
      <td>92.78</td>
      <td>104.88</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">ap_fixed&lt;16, 6&gt;</th>
      <th>1</th>
      <td>1.78</td>
      <td>199.86</td>
      <td>45.96</td>
      <td>184.86</td>
      <td>66.59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.30</td>
      <td>198.30</td>
      <td>45.71</td>
      <td>190.51</td>
      <td>68.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.38</td>
      <td>198.50</td>
      <td>45.95</td>
      <td>195.05</td>
      <td>73.15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.48</td>
      <td>175.18</td>
      <td>46.42</td>
      <td>188.65</td>
      <td>95.70</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.90</td>
      <td>83.85</td>
      <td>48.13</td>
      <td>184.96</td>
      <td>101.44</td>
    </tr>
    <tr>
      <th>32</th>
      <td>4.43</td>
      <td>51.04</td>
      <td>51.83</td>
      <td>193.38</td>
      <td>141.07</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.75</td>
      <td>30.32</td>
      <td>55.36</td>
      <td>193.26</td>
      <td>229.37</td>
    </tr>
  </tbody>
</table>

## Datasets
Training accurate predictors requires large datasets of synthesized neural networks. We used [hls4ml](https://github.com/fastmachinelearning/hls4ml) to synthesize neural networks generated with parameters randomly sampled from predefined ranges (defaults of data classes in the code). Our models' training data is publicly available at [https://borealisdata.ca/dataverse/rule4ml](https://borealisdata.ca/dataverse/rule4ml).

## Limitations
In their current iteration, the predictors can process [Keras](https://keras.io/about/) or [PyTorch](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) models to generate FPGA resources (**BRAM**, **DSP**, **FF**, **LUT**) and latency (**Clock Cycles**) estimations for various synthesis configurations. However, the training models are limited to specific layers: **Dense/Linear**, **ReLU**, **Tanh**, **Sigmoid**, **Softmax**, **BatchNorm**, **Add**, **Concatenate**, and **Dropout**. They are also constrained by synthesis parameters, notably **clock_period** (10 ns) and **io_type** (io_parallel). Inputs outside these configurations may result in inaccurate predictions.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/LICENSE) file for details.
