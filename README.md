# Controlling for Biasing Signals in Images: Survival Predictions for Lung Cancer with Deep Learning

[![DOI](https://zenodo.org/badge/207314717.svg)](https://zenodo.org/badge/latestdoi/207314717)

This repository contains the necessary files to reproduce the results of paper
"Controlling for Biasing Signals in Images: Survival Predictions for Lung Cancer with Deep Learning"
by W.A.C. van Amsterdam, J.J.C. Verhoeff, P.A. de Jong, T. Leiner and M.J.C. Eijkemans

## Replicating the experiments

Please follow these steps to replicate the results as published.
The original python scripts are (somewhat) self-explanatory.
They do contain unused code that was useful during initial experiments, but was not used for the final publication

### Pre-processing

**see writing/preprocessingoverview.png** for a schematic overview of the required preprocessing

**NOTE** the package pylidc was used in the pre-processing scripts, but may rely on older python libraries

### Running the models

Use PyTorch v1.0 to run the models

To run:

- define a data-generating mechanism by creating a .csv file in experiments/sims
- create a 'setting' that combines the data-generating mechanism with the modeling approach
- run 'simulate_data.py' to generate observations
- run 'train.py' to run the experiments

### Further files

- model/base_model/params.json contains the hyperparameters that controls how train.py runs
