
This repository contains the modeling framework for quantifying cerebrospinal fluid (CSF) flow velocity from fMRI inflow signals.
Please see our paper which describes the methodology:

**Ashenagar et al., 2025**  
[Modeling dynamic inflow effects in fMRI to quantify cerebrospinal fluid flow](https://doi.org/10.1162/IMAG.a.9)

See the [v1.0 tag](https://github.com/baarbod/tofmodel/releases/tag/v1.0) for the version of the code used the paper.
v1.0 is uploaded for reference to the exact methods used in the paper.
However, please use the latest version for improved usability and broader compatibility.

<br />
We developed a forward model to simulate fMRI inflow signals based on a given input velocity timeseries. This repository contains the forward model as well as a pipeline which generates a synthetic dataset containing simulated inflow signals which is used to train a convolutional neural network, enabling inversion of the forward model. The resulting neural network can predict a velocity timeseries given an inflow signal timeseries across multiple imaging slices. 

## Installation

cd into the directory and run: 
```bash
pip install -e .
```
> **Note:** An environment file will be added later to help get all required packages.

## Usage

### Steps:
1) Configure parameters in the config file
2) Run pipeline for synthetic dataset generation
3) Train neural network on synthetic dataset

#### Step 1 - CONFIGURATION

An example configuration file is provided (config.yml). There are groups of parameters that will be described below. <br />
**scan_param**: Parameters of the scanner. This should be identical to what was used when acquiring the fMRI data that will later on be used for inferring velocities. <br />
**data_simulation**: Control dimensions of the synthetic dataset and how it is batched. <br />
**sampling**: Define parameters for sampling the forward model input data. Here you control the flow dynamics and cross-sectional areas included in the dataset. <br />
**paths**: Paths you want to use.

#### Step 2 - RUNNING PIPELINE

A command-line tool (tofmodel/cli.py) is provided to faciliate running the pipeline locally or using job scripts (recommended when generating large datasets). <br /> 

For dataset generation, you can run each step as shown below:
```bash
tof inverse --config config.yml --mode sequential --action prepare_inputs
tof inverse --config config.yml --action sort_inputs
tof inverse --config config.yml --mode sequential --action run_simulations
tof inverse --config config.yml --action combine_simulations
tof inverse --config config.yml --action cleanup_directories
```
Or you can run all the steps with one command:
```bash
tof inverse --config config.yml --action run_all
```
```prepare_inputs``` and ```run_simulations``` are excecuted in batches. <br />```--mode sequential``` 
will run all batches in a loop locally. <br />```--mode singletask --taskid $ID``` 
will run only a particular batch to allow for passing the taskid via a job scheduler. If using a scheduler, set up scripts to run each step individually and wait for each step to finish before moving to the next.

#### Step 3 - TRAINING THE NETWORK
```bash
tof train --epochs $NUM_EPOCH --batch $BATCH_SIZE --lr $LEARNING_RATE \
          --noise_method $NOISE_METHOD --noise_scale $NOISE_SCALE --exp_name $NAME --config config.yml
```
## EXAMPLE

> **Note:** UNDER CONSTRUCTION




<br /> <br />
> **Note:** The codebase is currently being updated to enhance the pipeline and ensure compatibility with other computing environments. An updated readme will be made to illustrate example usage and detailed features of this repository.