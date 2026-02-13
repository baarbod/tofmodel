
This repository contains the modeling framework for quantifying cerebrospinal fluid (CSF) flow velocity from fMRI inflow signals.
Please see our paper which describes the methodology:

**Ashenagar et al., 2025**  
[Modeling dynamic inflow effects in fMRI to quantify cerebrospinal fluid flow](https://doi.org/10.1162/IMAG.a.9)

See the [v1.0 tag](https://github.com/baarbod/tofmodel/releases/tag/v1.0) for the version of the code used the paper.
v1.0 is uploaded for reference to the exact methods used in the paper.
However, please use the latest version for improved usability and broader compatibility.


NOTE:

The inverse module for this project is being superseded by my other repository (tofinv) which will implement an end-to-end automated pipeline for extracting velocities from the fMRI data.
The inverse module in this repository was our older implementation and was limited in that it required some manual tweaking and would be somewhat tedious to use.

Please see my other repository "tofinv" for using the pipeline.

The inverse module here will probably be removed as features are ported to tofinv, and in the future this will be where the forward model implementation is kept.
