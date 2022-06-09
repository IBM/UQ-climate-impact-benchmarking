<<<<<<< HEAD
# A Surrogate Model Benchmark for Dynamic Climate Impact Models

This repository accompanies our paper submitted to NeurIPS 2022 Datasets and Benchmarks track. 

**Abstract:** 

**Contact:** Julian Kuehnert [(julian [dot] kuehnert [at] ibm [dot] com)](mailto:julian.kuehnert@ibm.com)


## Overview:

* ``datasets/``: Files containing seasonal weather forecasts for precipitation and temperature.
* ``figures/``: Folder in which figures from results visualization are saved.
* ``models/``: Python scripts containing the R0 malaria model as well as the ML surrogate models (BLSTM and RFQR).
* ``trained_models/``: Folder in which trained ML models are saved.
* ``utils/``: Python scripts for evaluation and visualization of results.
* ``malaria_modeling.ipynb``: Python notebook for simulating the precipitation- and temperature-driven malaria transmission coefficient `R0`.
* ``ensemble_modeling.ipynb``: Python notebook for training surrogate models to predict uncertainties from ensemble weather forecasts. 

## Content:

### Dataset
Seasonal weather forecasts of precipitation and temperature for coordinate in Nairobi, Kenya:
![precipitation](figures/precipitation_2021.png)
![temperature](figures/temperature_2021.png)

### Models
Modeled malaria transmission coefficient R0, simulated using the climate-driven R0 model:
![R0](figures/R0_2021.png)

## Running Benchmarks

### Prerequisites and Python modules

Here we assume python3.9 and mac/linux, which are not necessarily mandatory.

    $ pip install jupyter numpy
    $ pip install -r requirements.txt

### Generate R0 for malaria model variables

Run through:

    Malaria_modeling.ipynb

for dataset preparation.  This will generate the ground truth data:

    datasets/R0_malaria_model_variables.csv

from

    datasets/seasonal_forecasts.csv

### Surrogate modeling

Surrogate Ensemble Modeling for Dynamic Climate Impact Models can be run as:

    ensemble_modeling.ipynb

Enjoy!

## License: 


## Citation:
