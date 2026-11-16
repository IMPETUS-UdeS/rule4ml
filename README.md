[![License](https://img.shields.io/badge/License-GPL_3.0-red.svg)](https://opensource.org/license/gpl-3-0)
[![PyPI version](https://badge.fury.io/py/rule4ml.svg)](https://badge.fury.io/py/rule4ml)

# rule4ml: Resource Utilization and Latency Estimation for ML

`rule4ml` is a tool designed for pre-synthesis estimation of FPGA resource utilization and inference latency for machine learning models.

## Installation

`rule4ml` releases are uploaded to the Python Package Index for easy installation via `pip`.

```bash
pip install rule4ml
```

This will only install the [base package](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/rule4ml) and its dependencies for resources and latency prediction. The [data_gen](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/data_gen/) scripts and the [Jupyter notebooks](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/notebooks) are to be cloned from the repo if needed.

The package includes only the latest predictors. To use the predictors of an earlier release, install that release, for example `pip install rule4ml==0.2.0`.

The base package uses PyTorch for the default predictors. Keras models and predictors need an extra:

```bash
pip install "rule4ml[keras]"
```

The data generation dependencies are listed separately in [data_gen/requirements.txt](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/data_gen/requirements.txt):

```bash
pip install -r data_gen/requirements.txt
```

## Getting Started

### Tutorial
To get started with `rule4ml`, please refer to the detailed Jupyter Notebook [tutorial](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/notebooks/tutorial.ipynb). This tutorial covers:

- Using pre-trained estimators for resources and latency predictions.
- Generating synthetic datasets.
- Training and testing your own predictors.

### Usage
The following example uses a Keras input model and requires `rule4ml[keras]`:

```python
import keras
from keras.layers import Input, Dense, Activation

from rule4ml.surrogates import MultiSurrogate

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
estimator = MultiSurrogate()
estimator.load_default_surrogates()

# MultiSurrogate predictions are formatted as a pandas DataFrame
prediction_df = estimator.predict(model_to_predict)

# Further formatting can be applied to organize the DataFrame
if not prediction_df.empty:
    prediction_df = prediction_df.set_index(
        [
            "Model",
            "Board",
            "Strategy",
            "Precision",
            "Reuse Factor",
            "Clock Period",
            "HLS4ML Version",
            "Vivado Version",
        ]
    )

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
      <th></th>
      <th></th>
      <th></th>
      <th>BRAM_36K</th>
      <th>BRAM_36K (%)</th>
      <th>DSP</th>
      <th>DSP (%)</th>
      <th>FF</th>
      <th>FF (%)</th>
      <th>LUT</th>
      <th>LUT (%)</th>
      <th>CYCLES_MIN</th>
      <th>CYCLES_MAX</th>
      <th>INTERVAL_MIN</th>
      <th>INTERVAL_MAX</th>
    </tr>
    <tr>
      <th>Model</th>
      <th>Board</th>
      <th>Strategy</th>
      <th>Precision</th>
      <th>Reuse Factor</th>
      <th>Clock Period</th>
      <th>HLS4ML Version</th>
      <th>Vivado Version</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2016" valign="top">Jet Classifier</th>
      <th rowspan="504" valign="top">pynq-z2</th>
      <th rowspan="252" valign="top">latency</th>
      <th rowspan="84" valign="top">ap_fixed&lt;2, 1&gt;</th>
      <th rowspan="12" valign="top">1</th>
      <th rowspan="6" valign="top">5.0</th>
      <th rowspan="3" valign="top">0.8.1</th>
      <th>2019.1</th>
      <td>0.03</td>
      <td>0.02</td>
      <td>4.29</td>
      <td>1.95</td>
      <td>599.50</td>
      <td>0.56</td>
      <td>980.93</td>
      <td>1.84</td>
      <td>53.85</td>
      <td>46.26</td>
      <td>1.30</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.03</td>
      <td>0.02</td>
      <td>5.48</td>
      <td>2.49</td>
      <td>701.46</td>
      <td>0.66</td>
      <td>1477.54</td>
      <td>2.78</td>
      <td>20.59</td>
      <td>18.50</td>
      <td>1.08</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.08</td>
      <td>0.05</td>
      <td>5.57</td>
      <td>2.53</td>
      <td>1350.86</td>
      <td>1.27</td>
      <td>3627.03</td>
      <td>6.82</td>
      <td>17.19</td>
      <td>13.91</td>
      <td>1.03</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1.1.0</th>
      <th>2019.1</th>
      <td>0.11</td>
      <td>0.08</td>
      <td>4.53</td>
      <td>2.06</td>
      <td>979.96</td>
      <td>0.92</td>
      <td>2149.31</td>
      <td>4.04</td>
      <td>50.49</td>
      <td>43.05</td>
      <td>1.07</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.09</td>
      <td>0.06</td>
      <td>5.30</td>
      <td>2.41</td>
      <td>1566.92</td>
      <td>1.47</td>
      <td>5547.77</td>
      <td>10.43</td>
      <td>18.70</td>
      <td>15.88</td>
      <td>0.94</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.52</td>
      <td>2.51</td>
      <td>2930.06</td>
      <td>2.75</td>
      <td>11429.28</td>
      <td>21.48</td>
      <td>15.36</td>
      <td>12.14</td>
      <td>0.91</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">10.0</th>
      <th rowspan="3" valign="top">0.8.1</th>
      <th>2019.1</th>
      <td>0.03</td>
      <td>0.02</td>
      <td>4.11</td>
      <td>1.87</td>
      <td>547.23</td>
      <td>0.51</td>
      <td>905.81</td>
      <td>1.70</td>
      <td>54.16</td>
      <td>46.58</td>
      <td>1.28</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.04</td>
      <td>0.03</td>
      <td>5.60</td>
      <td>2.55</td>
      <td>633.04</td>
      <td>0.59</td>
      <td>1286.51</td>
      <td>2.42</td>
      <td>20.78</td>
      <td>18.52</td>
      <td>1.08</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>5.65</td>
      <td>2.57</td>
      <td>1102.98</td>
      <td>1.04</td>
      <td>2929.63</td>
      <td>5.51</td>
      <td>17.12</td>
      <td>13.82</td>
      <td>1.03</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1.1.0</th>
      <th>2019.1</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>4.24</td>
      <td>1.93</td>
      <td>843.36</td>
      <td>0.79</td>
      <td>1764.27</td>
      <td>3.32</td>
      <td>50.88</td>
      <td>43.35</td>
      <td>1.07</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>5.30</td>
      <td>2.41</td>
      <td>1293.30</td>
      <td>1.22</td>
      <td>4422.43</td>
      <td>8.31</td>
      <td>18.72</td>
      <td>15.78</td>
      <td>0.95</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.42</td>
      <td>2.46</td>
      <td>2589.22</td>
      <td>2.43</td>
      <td>9456.45</td>
      <td>17.78</td>
      <td>15.21</td>
      <td>12.09</td>
      <td>0.92</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">2</th>
      <th rowspan="6" valign="top">5.0</th>
      <th rowspan="3" valign="top">0.8.1</th>
      <th>2019.1</th>
      <td>0.02</td>
      <td>0.01</td>
      <td>4.30</td>
      <td>1.96</td>
      <td>579.08</td>
      <td>0.54</td>
      <td>1021.74</td>
      <td>1.92</td>
      <td>54.99</td>
      <td>49.04</td>
      <td>1.69</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.02</td>
      <td>0.02</td>
      <td>5.73</td>
      <td>2.60</td>
      <td>675.12</td>
      <td>0.63</td>
      <td>1592.40</td>
      <td>2.99</td>
      <td>23.35</td>
      <td>19.90</td>
      <td>1.40</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.05</td>
      <td>0.04</td>
      <td>5.72</td>
      <td>2.60</td>
      <td>1344.83</td>
      <td>1.26</td>
      <td>3781.88</td>
      <td>7.11</td>
      <td>18.38</td>
      <td>15.25</td>
      <td>1.28</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1.1.0</th>
      <th>2019.1</th>
      <td>0.08</td>
      <td>0.06</td>
      <td>4.59</td>
      <td>2.08</td>
      <td>941.90</td>
      <td>0.89</td>
      <td>2247.90</td>
      <td>4.23</td>
      <td>51.75</td>
      <td>45.21</td>
      <td>1.40</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.08</td>
      <td>0.06</td>
      <td>5.43</td>
      <td>2.47</td>
      <td>1552.81</td>
      <td>1.46</td>
      <td>5759.89</td>
      <td>10.83</td>
      <td>19.90</td>
      <td>16.95</td>
      <td>1.10</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.68</td>
      <td>2.58</td>
      <td>2868.36</td>
      <td>2.70</td>
      <td>11879.30</td>
      <td>22.33</td>
      <td>16.91</td>
      <td>13.35</td>
      <td>1.04</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">10.0</th>
      <th rowspan="3" valign="top">0.8.1</th>
      <th>2019.1</th>
      <td>0.01</td>
      <td>0.01</td>
      <td>4.09</td>
      <td>1.86</td>
      <td>527.72</td>
      <td>0.50</td>
      <td>942.59</td>
      <td>1.77</td>
      <td>55.30</td>
      <td>49.42</td>
      <td>1.66</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.01</td>
      <td>0.01</td>
      <td>5.80</td>
      <td>2.64</td>
      <td>611.82</td>
      <td>0.58</td>
      <td>1373.46</td>
      <td>2.58</td>
      <td>23.68</td>
      <td>20.06</td>
      <td>1.39</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.04</td>
      <td>0.03</td>
      <td>5.89</td>
      <td>2.68</td>
      <td>1088.38</td>
      <td>1.02</td>
      <td>3074.15</td>
      <td>5.78</td>
      <td>18.34</td>
      <td>15.17</td>
      <td>1.27</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1.1.0</th>
      <th>2019.1</th>
      <td>0.05</td>
      <td>0.04</td>
      <td>4.24</td>
      <td>1.93</td>
      <td>808.14</td>
      <td>0.76</td>
      <td>1863.84</td>
      <td>3.50</td>
      <td>52.09</td>
      <td>45.56</td>
      <td>1.37</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>0.06</td>
      <td>0.04</td>
      <td>5.56</td>
      <td>2.53</td>
      <td>1270.70</td>
      <td>1.19</td>
      <td>4610.81</td>
      <td>8.67</td>
      <td>20.03</td>
      <td>16.94</td>
      <td>1.10</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.56</td>
      <td>2.53</td>
      <td>2587.28</td>
      <td>2.43</td>
      <td>9794.78</td>
      <td>18.41</td>
      <td>16.79</td>
      <td>13.25</td>
      <td>1.04</td>
      <td>1.13</td>
    </tr>
  </tbody>
</table>

## Datasets
Training accurate predictors requires large datasets of synthesized neural networks. We used [hls4ml](https://github.com/fastmachinelearning/hls4ml) to synthesize neural networks generated with parameters randomly sampled from predefined ranges (defaults of data classes in the code). Our models' training data is publicly available at [https://borealisdata.ca/dataverse/rule4ml](https://borealisdata.ca/dataverse/rule4ml).

Newer predictors were trained on `wa-hls4ml`, a bigger dataset including more architectures and parameter ranges. This dataset, along with the HLS project files, can be found at [https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml) and [https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml-projects](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml-projects).

## Accuracy
The table shows errors of the default Transformer surrogates in distribution (in), on the wa-hls4ml test split built with the same generators as training, and out of distribution (out), on the wa-hls4ml exemplar models, seven hand-built model families absent from training. RMSE is in the target's units: BRAM_36K blocks, DSP, FF and LUT counts, and clock cycles.

| Target | SMAPE (in) | RMSE (in) | R² (in) | SMAPE (out) | RMSE (out) | R² (out) |
|---|---|---|---|---|---|---|
| BRAM | 4.0% | 2.5 | 0.973 | 89.6% | 1.8 | 0.459 |
| DSP | 0.5% | 61 | 0.990 | 45.6% | 250 | 0.906 |
| FF | 1.0% | 3,094 | 0.992 | 41.9% | 5,573 | 0.892 |
| LUT | 2.7% | 6,159 | 0.988 | 23.2% | 10,370 | 0.857 |
| Cycles min | 0.9% | 62,920 | 0.992 | 48.7% | 355 | 0.920 |
| Cycles max | 0.9% | 51,790 | 0.994 | 50.1% | 319 | 0.936 |
| Interval min | 2.6% | 1,174 | 0.948 | 43.5% | 333 | 0.572 |
| Interval max | 2.8% | 111,700 | 0.974 | 46.1% | 296 | 0.662 |

## Limitations
The default predictors take Keras, QKeras or PyTorch models and estimate FPGA resources (**BRAM**, **DSP**, **FF**, **LUT**) and latency (minimum and maximum **clock cycles** and **interval**). Inputs outside their training data may give inaccurate predictions. Settings absent from training raise a warning.

- **Layers:** Dense/Linear, Conv1D and Conv2D, max and average pooling, Flatten, BatchNorm, Add, Concatenate, Dropout, and the ReLU, Sigmoid, Tanh and Softmax activations. The `wa-hls4ml` training data has no Add or Concatenate layers and at most 55 layers per model.
- **Configurations:** the pynq-z2, zcu102, alveo-u200 and alveo-u250 boards (mostly alveo-u250 in training), the Latency and Resource strategies, hls4ml 0.8.1 and 1.1.0, and Vivado 2019.1, 2023.2 and 2024.2. Other versions are accepted with a warning and add no version information.
- **Synthesis settings:** the clock period comes from the config's `ClockPeriod`, else the hls4ml default of 5 ns. Training used 5 and 10 ns, and 10 ns only with Vivado 2019.1, so the effect of the clock is weakly learned. Convolutional training models used `io_stream` and dense ones `io_parallel`, and the predictors assume the same. Other combinations raise a warning.
- **Outputs:** predictions are not rounded, so counts such as BRAM_36K and LUT can be fractional. Each target has its own predictor, so a predicted minimum can exceed its maximum.

## Citation
If you use rule4ml in a publication, please cite:
```bibtex
@article{rahimifar2025rule4ml,
  author    = {Rahimifar, Mohammad Mehdi and Ezzaoui Rahali, Hamza and Therrien, Audrey C.},
  title     = {{rule4ml: an open-source tool for resource utilization and latency estimation for ML models on FPGA}},
  journal   = {Machine Learning: Science and Technology},
  publisher = {IOP Publishing},
  volume    = {6},
  number    = {1},
  pages     = {015009},
  year      = {2025},
  doi       = {10.1088/2632-2153/ada71c}
}
```
If you use the predictors trained on `wa-hls4ml`, please also cite:
```bibtex
@article{hawks2026wahls4ml,
  author    = {Hawks, Benjamin and Weitz, Jason and Demler, Dmitri and Tame-Narvaez, Karla and Plotnikov, Dennis and Rahimifar, Mohammad Mehdi and Ezzaoui Rahali, Hamza and Therrien, Audrey C. and Sproule, Donovan and Khoda, Elham E and Smith, Keegan A. and Marroquin, Russell and Di Guglielmo, Giuseppe and Tran, Nhan and Duarte, Javier and Loncar, Vladimir},
  title     = {{wa-hls4ml: A Benchmark and Surrogate Models for hls4ml Resource and Latency Estimation}},
  journal   = {ACM Transactions on Reconfigurable Technology and Systems},
  publisher = {Association for Computing Machinery},
  volume    = {19},
  number    = {2},
  pages     = {20},
  year      = {2026},
  doi       = {10.1145/3787490}
}
```

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/IMPETUS-UdeS/rule4ml/tree/main/LICENSE) file for details.
