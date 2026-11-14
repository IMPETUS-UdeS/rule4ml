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
      <th>BRAM_36K</th>
      <th>BRAM_36K (%)</th>
      <th>DSP</th>
      <th>DSP (%)</th>
      <th>FF</th>
      <th>FF (%)</th>
      <th>LUT</th>
      <th>LUT (%)</th>
      <th>CYCLES</th>
      <th>INTERVAL</th>
    </tr>
    <tr>
      <th>Model</th>
      <th>Board</th>
      <th>Strategy</th>
      <th>Precision</th>
      <th>Reuse Factor</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4032" valign="top">Jet Classifier</th>
      <th rowspan="1008" valign="top">pynq-z2</th>
      <th rowspan="504" valign="top">latency</th>
      <th rowspan="168" valign="top">ap_fixed&lt;2, 1&gt;</th>
      <th rowspan="24" valign="top">1</th>
      <th rowspan="12" valign="top">0.8.1</th>
      <th>2019.1</th>
      <td>2.52</td>
      <td>1.80</td>
      <td>0.32</td>
      <td>0.14</td>
      <td>1265.02</td>
      <td>1.19</td>
      <td>3564.90</td>
      <td>6.70</td>
      <td>125.77</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2019.2</th>
      <td>2.47</td>
      <td>1.76</td>
      <td>0.48</td>
      <td>0.22</td>
      <td>1262.29</td>
      <td>1.19</td>
      <td>3380.57</td>
      <td>6.35</td>
      <td>115.48</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2020.1</th>
      <td>2.29</td>
      <td>1.64</td>
      <td>0.49</td>
      <td>0.22</td>
      <td>1109.34</td>
      <td>1.04</td>
      <td>3279.37</td>
      <td>6.16</td>
      <td>115.62</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2020.2</th>
      <td>2.55</td>
      <td>1.82</td>
      <td>0.53</td>
      <td>0.24</td>
      <td>1490.04</td>
      <td>1.40</td>
      <td>3457.23</td>
      <td>6.50</td>
      <td>118.07</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2021.1</th>
      <td>2.31</td>
      <td>1.65</td>
      <td>0.44</td>
      <td>0.20</td>
      <td>1054.50</td>
      <td>0.99</td>
      <td>2915.67</td>
      <td>5.48</td>
      <td>118.99</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2021.2</th>
      <td>2.48</td>
      <td>1.77</td>
      <td>0.58</td>
      <td>0.26</td>
      <td>1085.17</td>
      <td>1.02</td>
      <td>3072.19</td>
      <td>5.77</td>
      <td>117.91</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2022.1</th>
      <td>2.53</td>
      <td>1.81</td>
      <td>0.47</td>
      <td>0.21</td>
      <td>1301.50</td>
      <td>1.22</td>
      <td>3093.67</td>
      <td>5.82</td>
      <td>119.36</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2022.2</th>
      <td>2.43</td>
      <td>1.74</td>
      <td>0.57</td>
      <td>0.26</td>
      <td>1150.09</td>
      <td>1.08</td>
      <td>3032.74</td>
      <td>5.70</td>
      <td>119.39</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2023.1</th>
      <td>2.51</td>
      <td>1.79</td>
      <td>0.59</td>
      <td>0.27</td>
      <td>1357.55</td>
      <td>1.28</td>
      <td>3327.19</td>
      <td>6.25</td>
      <td>118.30</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>2.39</td>
      <td>1.71</td>
      <td>0.29</td>
      <td>0.13</td>
      <td>304.04</td>
      <td>0.29</td>
      <td>2689.27</td>
      <td>5.06</td>
      <td>108.34</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2024.1</th>
      <td>2.41</td>
      <td>1.72</td>
      <td>0.54</td>
      <td>0.25</td>
      <td>1574.28</td>
      <td>1.48</td>
      <td>3517.61</td>
      <td>6.61</td>
      <td>116.26</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>2.08</td>
      <td>1.49</td>
      <td>0.77</td>
      <td>0.35</td>
      <td>936.16</td>
      <td>0.88</td>
      <td>2780.73</td>
      <td>5.23</td>
      <td>110.77</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">1.1.0</th>
      <th>2019.1</th>
      <td>2.57</td>
      <td>1.84</td>
      <td>1.16</td>
      <td>0.53</td>
      <td>1237.20</td>
      <td>1.16</td>
      <td>2434.88</td>
      <td>4.58</td>
      <td>37.70</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2019.2</th>
      <td>2.53</td>
      <td>1.81</td>
      <td>1.39</td>
      <td>0.63</td>
      <td>1273.41</td>
      <td>1.20</td>
      <td>2317.88</td>
      <td>4.36</td>
      <td>28.73</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2020.1</th>
      <td>2.35</td>
      <td>1.68</td>
      <td>1.42</td>
      <td>0.65</td>
      <td>1023.07</td>
      <td>0.96</td>
      <td>2275.59</td>
      <td>4.28</td>
      <td>28.97</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2020.2</th>
      <td>2.64</td>
      <td>1.89</td>
      <td>1.45</td>
      <td>0.66</td>
      <td>1314.61</td>
      <td>1.24</td>
      <td>2359.94</td>
      <td>4.44</td>
      <td>30.62</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2021.1</th>
      <td>2.34</td>
      <td>1.67</td>
      <td>1.35</td>
      <td>0.61</td>
      <td>983.35</td>
      <td>0.92</td>
      <td>2025.47</td>
      <td>3.81</td>
      <td>31.37</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2021.2</th>
      <td>2.56</td>
      <td>1.83</td>
      <td>1.50</td>
      <td>0.68</td>
      <td>1149.12</td>
      <td>1.08</td>
      <td>2167.54</td>
      <td>4.07</td>
      <td>30.66</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2022.1</th>
      <td>2.65</td>
      <td>1.89</td>
      <td>1.39</td>
      <td>0.63</td>
      <td>1104.21</td>
      <td>1.04</td>
      <td>2131.50</td>
      <td>4.01</td>
      <td>31.74</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2022.2</th>
      <td>2.47</td>
      <td>1.76</td>
      <td>1.49</td>
      <td>0.68</td>
      <td>1200.66</td>
      <td>1.13</td>
      <td>2120.53</td>
      <td>3.99</td>
      <td>31.79</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2023.1</th>
      <td>2.58</td>
      <td>1.84</td>
      <td>1.64</td>
      <td>0.74</td>
      <td>1247.67</td>
      <td>1.17</td>
      <td>2301.45</td>
      <td>4.33</td>
      <td>30.79</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2023.2</th>
      <td>2.49</td>
      <td>1.78</td>
      <td>1.14</td>
      <td>0.52</td>
      <td>499.64</td>
      <td>0.47</td>
      <td>1795.66</td>
      <td>3.38</td>
      <td>25.01</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2024.1</th>
      <td>2.46</td>
      <td>1.76</td>
      <td>1.45</td>
      <td>0.66</td>
      <td>1373.96</td>
      <td>1.29</td>
      <td>2405.98</td>
      <td>4.52</td>
      <td>29.38</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2024.2</th>
      <td>2.09</td>
      <td>1.49</td>
      <td>1.99</td>
      <td>0.91</td>
      <td>1059.89</td>
      <td>1.00</td>
      <td>2089.47</td>
      <td>3.93</td>
      <td>26.71</td>
      <td>1.35</td>
    </tr>
  </tbody>
</table>

## Datasets
Training accurate predictors requires large datasets of synthesized neural networks. We used [hls4ml](https://github.com/fastmachinelearning/hls4ml) to synthesize neural networks generated with parameters randomly sampled from predefined ranges (defaults of data classes in the code). Our models' training data is publicly available at [https://borealisdata.ca/dataverse/rule4ml](https://borealisdata.ca/dataverse/rule4ml).

Newer predictors were trained on `wa-hls4ml`, a bigger dataset including more architectures and parameter ranges. This dataset, along with the HLS project files, can be found at [https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml) and [https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml-projects](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml-projects).

## Limitations
The default predictors take Keras, QKeras or PyTorch models and estimate FPGA resources (**BRAM**, **DSP**, **FF**, **LUT**) and latency (**Clock Cycles**, **Interval**). Inputs outside their training data may give inaccurate predictions.

- **Layers:** Dense/Linear, Conv1D and Conv2D, max and average pooling, Flatten, BatchNorm, Add, Concatenate, Dropout, and the ReLU, Sigmoid, Tanh and Softmax activations.
- **Configurations:** the pynq-z2, zcu102, alveo-u200 and alveo-u250 boards, the Latency and Resource strategies, hls4ml 0.8.1 and 1.1.0, and Vivado 2019.1 to 2024.2.
- **Synthesis settings:** clock period and io_type are not model inputs. Most training records use a 5 or a 10 ns clock with io_parallel.
- **Latency:** v1 and v2 predict the average of the minimum and maximum. For convolutional models the minimum and maximum interval can differ by orders of magnitude, so their interval estimate is rough.

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
