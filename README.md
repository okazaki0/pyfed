# PyFed

This is the code accompanying the paper "PyFed: Exteding PySyft with N-IID Federated Learning Benchmark" Paper link:  (Coming soon)

## About

PyFed is a benchmarking framework for federated learning extending [PySyft](https://github.com/OpenMined/PySyft), in a generic and distributed way. PyFed supports different aggregations methods and data distributions (Independent and Identically Distributed Data (IID) and Non-IID).

In this sense, PyFed is an alternative benchemarking framework of [LEAF](https://github.com/TalwalkarLab/leaf) for Federated Learning for PySyft.

The benchmarking is done using five dataset: `mnist`, `fashionmnist`, `cifar10`, `sent140`, `shakespeare`.

## Table of Contents
- [PyFed](#pyfed)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
    + [Depdendencies](#dependencies)
    + [Install Dependencies](install-dependencies)
  * [Usage](#usage)
    + [Launch Workers](#launch-the-workers)
    + [Launch Training](#launch-the-training)
  * [Results](#results)
  * [Contributing](#contributing)

## Installation

### Dependencies

Tested stable dependencies:

- [PySyft](https://github.com/OpenMined/PySyft) v0.2.5
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) v1.4.0

### Install Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements of PyFed.

```bash
pip install -r requirements.txt
```

## Usage

For running PyFed, please follow the next steps:
1. Launch the workers: `python run/network/start_websocket_server.py [arguments]`
2. Launch the training: `python run/network/main.py [arguments]`
3. Get the results

All arguments have default values. However, these arguments should be set to the desired settings. 

### Launch the Workers

Workers can be launched using different arguments (see below)

#### Arguments

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `clients`                     | The number of clients. |
| `dataset`      | Dataset to be used. `mnist`, `fashionmnist`, `cifar10`, `sent140`, `shakespeare`. |
| `split_mode` | The split mode used either `iid` or `niid`. |
| `global_dataset` | Share global dataset over all clients. |
| `data_rate` | Percentage of samples in the global dataset to be added. |
| `add_error` | Add error to some samples either `True` or `False`. |
| `error_rate` | Percentage of error to be added. |

In the case of IID distribution (`split_mode = iid`), the following agruments are available:

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `iid_share`                     | Share samples between clients in the iid split mode. |
| `iid_rate`      | Percentage of samples to share between clients|

In the case of Non-IID distribution (`split_mode = niid`), the following agruments are available:

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `data_size`                     | The number of samples that hold each client. |
| `type`      | `random` split or `label` split using labels. |
| `label_num` | The number of classes holded by a client when with `label` split type. |
| `share_samples` |  How to share samples between clients holding the same classes. In the case of `label` split type, the following values are possible :<ul><li>0: clients holding the same class share also the same samples</li><li>1: clients holding the same class might also share the same samples (random sampling)</li><li>2: clients holding the same class have different samples from this class</li><li>3: Share different class (sum(label_num) must <= number of class)</li></ul> |

#### Example

There is two ways to define arguments: manually or using a config file. 

**Manually**

```
python run/network/start_websocket_server.py --clients = 5 /
--dataset = mnist /
--split_mode = niid /
--type = label /
--data_size = [234,2134,64,4132,1000] /
--label_num = [3,8,5,2,3] /
--share_samples = 2
```

Or using **config.yaml** file in `utils/`

```
python run/network/start_websocket_server.py --config_file True
```

### Launch the training

After launching the workers correctly, we are ready to start the training using the following arguments.

#### Arguments

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model that we will use e.g.`cnn`,`lstm`. |
| `batch_size`      | The batch size of the training. |
| `test_batch_size` | The batch size used for the test data. |
| `training_rounds` | The number of federated learning rounds. |
| `federate_after_n_batches` | The number of training steps performed on each remote worker before averaging. |
| `lr` | The learning rate. |
| `cuda` | The use cuda. |
| `seed` | The seed used for randomization. |
| `eval_every` | Evaluate the model evrey n rounds. |
| `fraction_client` | The number of clients that will in each round. |
| `optimizer` | The optimazer that we will use : `SGD` or `Adam`. |
| `aggregation` | The type of aggragation : `federated_avg` or `wieghted_avg`. |
| `loss` | The loss function : nll_loss or cross_entropy. |

#### Example

```
python run/network/main.py –config_file True
```

### Results

Here is some experimentation results that we get using our framework, you can check all the results and the configuration in the experimentation package.

<table width="110%">
<tbody>
<tr>
<td rowspan="3" width="8%" text-align: center;>
<p><strong>Dataset</strong></p>
</td>
<td rowspan="3" width="5%" text-align: center;>
<p><strong>Model</strong></p>
</td>
<td colspan="2" rowspan="2" width="12%" text-align: center;>
<p><strong>Base line (iid)</strong></p>
</td>
<td colspan="6" width="37%" text-align: center;>
<p><strong>Non iid (split by label)</strong></p>
</td>
<td colspan="2" rowspan="2" width="12%" text-align: center;>
<p><strong>Non iid (random split)</strong></p>
</td>
<td rowspan="3" width="5%" text-align: center;>
<p><strong>Epochs</strong></p>
</td>
<td rowspan="3" width="8%" text-align: center;>
<p><strong>Bach-size</strong></p>
</td>
<td rowspan="3" width="5%" text-align: center;>
<p><strong>fraction</strong></p>
</td>
<td rowspan="3" width="3%" text-align: center;>
<p><strong>lr</strong></p>
</td>
<td rowspan="3" width="5%" text-align: center;>
<p><strong>Rounds</strong></p>
</td>
</tr>
<tr>
<td colspan="2" width="12%" text-align: center;>
<p><strong>Type 0</strong></p>
</td>
<td colspan="2" width="12%" text-align: center;>
<p><strong>Type 1</strong></p>
</td>
<td colspan="2" width="12%" text-align: center;>
<p><strong>Type 2</strong></p>
</td>
</tr>
<tr>
<td width="6%" text-align: center;>
<p><strong>Accuracy</strong></p>
</td>
<td width="5%" text-align: center;>
<p><strong>Loss</strong></p>
</td>
<td width="6%" text-align: center;>
<p><strong>Accuracy</strong></p>
</td>
<td width="5%" text-align: center;>
<p><strong>Loss</strong></p>
</td>
<td width="6%" text-align: center;>
<p><strong>Accuracy</strong></p>
</td>
<td width="5%" text-align: center;>
<p><strong>Loss</strong></p>
</td>
<td width="6%" text-align: center;>
<p><strong>Accuracy</strong></p>
</td>
<td width="5%" text-align: center;>
<p><strong>Loss</strong></p>
</td>
<td width="6%" text-align: center;>
<p><strong>Accuracy</strong></p>
</td>
<td width="5%" text-align: center;>
<p><strong>Loss</strong></p>
</td>
</tr>
<tr>
<td width="8%" text-align: center;>
<p>Cifar10</p>
</td>
<td width="5%">
<p>CNN</p>
</td>
<td width="6%">
<p>67</p>
</td>
<td width="5%">
<p>0.8043</p>
</td>
<td width="6%">
<p>66.78</p>
</td>
<td width="5%">
<p>0.8132</p>
</td>
<td width="7%">
<p>65.89</p>
</td>
<td width="5%">
<p>0.8453</p>
</td>
<td width="6%">
<p>65.45</p>
</td>
<td width="6%">
<p>0.8464</p>
</td>
<td width="6%">
<p>66.89</p>
</td>
<td width="5%">
<p>0.8121</p>
</td>
<td width="5%">
<p>1</p>
</td>
<td width="4%">
<p>5</p>
</td>
<td width="5%">
<p>0.1</p>
</td>
<td width="3%">
<p>0.1</p>
</td>
<td width="5%">
<p>2500</p>
</td>
</tr>
<tr>
<td width="8%">
<p>Fashionmnist</p>
</td>
<td width="5%">
<p>CNN</p>
</td>
<td width="6%">
<p>86.81</p>
</td>
<td width="5%">
<p>0.368</p>
</td>
<td width="6%">
<p>85.36</p>
</td>
<td width="5%">
<p>0.4029</p>
</td>
<td width="7%">
<p>85.8</p>
</td>
<td width="5%">
<p>0.3956</p>
</td>
<td width="6%">
<p>85.42</p>
</td>
<td width="6%">
<p>0.4009</p>
</td>
<td width="6%">
<p>86.57</p>
</td>
<td width="5%">
<p>0.3727</p>
</td>
<td width="5%">
<p>1</p>
</td>
<td width="4%">
<p>10</p>
</td>
<td width="5%">
<p>0.1</p>
</td>
<td width="3%">
<p>0.1</p>
</td>
<td width="5%">
<p>100</p>
</td>
</tr>
<tr>
<td rowspan="2" width="8%">
<p>Mnist</p>
</td>
<td width="5%">
<p>CNN</p>
</td>
<td width="6%">
<p>95.63</p>
</td>
<td width="5%">
<p>0.1384</p>
</td>
<td width="6%">
<p>93.45</p>
</td>
<td width="5%">
<p>0.2171</p>
</td>
<td width="7%">
<p>93.88</p>
</td>
<td width="5%">
<p>0.2164</p>
</td>
<td width="6%">
<p>93.84</p>
</td>
<td width="6%">
<p>0.2086</p>
</td>
<td width="6%">
<p>95.04</p>
</td>
<td width="5%">
<p>0.1671</p>
</td>
<td width="5%">
<p>1</p>
</td>
<td width="4%">
<p>10</p>
</td>
<td width="5%">
<p>0.1</p>
</td>
<td width="3%">
<p>0.1</p>
</td>
<td width="5%">
<p>20</p>
</td>
</tr>
<tr>
<td width="5%">
<p>CNN</p>
<p><em>(with batch normalization)</em></p>
</td>
<td width="6%">
<p>96.33</p>
</td>
<td width="5%">
<p>0.1154</p>
</td>
<td width="6%">
<p>94.25</p>
</td>
<td width="5%">
<p>0.1902</p>
</td>
<td width="7%">
<p>94.74</p>
</td>
<td width="5%">
<p>0.1771</p>
</td>
<td width="6%">
<p>94.76</p>
</td>
<td width="6%">
<p>0.1884</p>
</td>
<td width="6%">
<p>96.09</p>
</td>
<td width="5%">
<p>0.13</p>
</td>
<td width="5%">
<p>1</p>
</td>
<td width="4%">
<p>10</p>
</td>
<td width="5%">
<p>0.1</p>
</td>
<td width="3%">
<p>0.1</p>
</td>
<td width="5%">
<p>20</p>
</td>
</tr>
<tr>
<td width="8%">
<p>Sent 140</p>

</td>
<td width="5%">
<p>LSTM</p>
</td>
<td width="6%">
<p>65.45</p>
</td>
<td width="5%">
<p>0.8345</p>
</td>
<td width="6%">
<p>64.4</p>
</td>
<td width="5%">
<p>0.9244</p>
</td>
<td width="7%">
<p>64.23</p>
</td>
<td width="5%">
<p>0.9445</p>
</td>
<td width="6%">
<p>65.78</p>
</td>
<td width="6%">
<p>0.8123</p>
</td>
<td width="6%">
<p>65.1</p>
</td>
<td width="5%">
<p>0.8663</p>
</td>
<td width="5%">
<p>1</p>
</td>
<td width="4%">
<p>1</p>
</td>
<td width="5%">
<p>0.1</p>
</td>
<td width="3%">
<p>0.1</p>
</td>
<td width="5%">
<p>1000</p>
</td>
</tr>
<tr>
<td width="8%">
<p>Shakespeare</p>
</td>
<td width="5%">
<p>GRU</p>
</td>
<td width="6%">
<p>50.36</p>
</td>
<td width="5%">
<p>1.2452</p>
</td>
<td width="6%">
<p>48.26</p>
</td>
<td width="5%">
<p>1.3452</p>
</td>
<td width="7%">
<p>48.76</p>
</td>
<td width="5%">
<p>1.2052</p>
</td>
<td width="6%">
<p>45.23</p>
</td>
<td width="6%">
<p>1.7452</p>
</td>
<td width="6%">
<p>49.46</p>
</td>
<td width="5%">
<p>1.2952</p>
</td>
<td width="5%">
<p>1</p>
</td>
<td width="4%">
<p>1</p>
</td>
<td width="5%">
<p>0.1</p>
</td>
<td width="3%">
<p>0.8</p>
</td>
<td width="5%">
<p>2000</p>
</td>
</tr>
</tbody>
</table>


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
