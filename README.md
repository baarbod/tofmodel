
This repository contains the modeling framework for simulating fMRI inflow signals from an input velocity timeseries.
Please see our paper which describes the methodology:

**Ashenagar et al., 2025**  
[Modeling dynamic inflow effects in fMRI to quantify cerebrospinal fluid flow](https://doi.org/10.1162/IMAG.a.9)

See the [v1.0 tag](https://github.com/baarbod/tofmodel/releases/tag/v1.0) for the version of the code used the paper.
v1.0 is uploaded for reference to the exact methods used in the paper.
However, please use the latest version for improved usability and broader compatibility.

The forward model in this repository is used in my other repository (tofinv). If you need to estimate velocity from inflow signals (i.e. inverse problem) please see tofinv.

See example_forward_model.py for details on usage of the forward model. 


### Installation

Create a new directory (optional but recommended)
```bash
mkdir -p repos
cd repos
```
If you already have a python environment, you can skip the next couple of steps. \
To create an isolated Python virtual environment, run the following command (replace `python3.10` with your specific path if needed):
```bash
python3.10 -m venv .venv
```
Then activate the environment
```bash
source .venv/bin/activate
``` 

Clone the repository
```bash
git clone https://github.com/baarbod/tofmodel.git
```
Navigate into the cloned repository
```bash
cd tofmodel
```
Install the package and all dependencies
```bash
pip install .
```


