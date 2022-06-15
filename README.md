# A Surrogate Model Benchmark for Dynamic Climate Impact Models

This repository accompanies our paper submitted to [NeurIPS 2022 Datasets and Benchmarks Track](https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks). 

**Abstract:** As acute climate change impacts weather and climate variability, there is an increasing need for robust climate impact model predictions that account for this variability and project it onto a range of potential climate hazard scenarios. The range of possible climate variables which form the input of climate impact models is typically represented by ensemble forecasts that capture the inherent uncertainty of weather models. To project the uncertainty associated with the distribution of input climate forecasts onto impact model output scenarios, each forecast ensemble member must be propagated through the physical model. In the case of complex impact models, this process is computationally expensive. It is therefore desirable to train an ML surrogate model to predict ensembles of climate hazard scenarios under reduced computational costs. To enable benchmarking for dynamic climate impact models, we release a dataset of seasonal weather forecasts, comprising 50 ensemble members of temperature and precipitation forecasts of 6-month horizon, spanning a period of 5 years for a single location in Nairobi, Kenya. This dataset is accompanied by a climate driven disease model, the Liverpool Malaria Model (LMM), which predicts the malaria transmission coefficient `R0`.  Two types of uncertainty-aware surrogate models are provided as reference implementations, namely a Random Forest Quantile Regression (RFQR) model and a Bayesian Long Short-Term Memory (BLSTM) neural network. The BLSTM is found to predict time series of individual ensemble members with higher accuracy and precision compared to the RFQR. By using the predicted confidence in each ensemble member, we can account for the uncertainty of previously unobserved weather conditions when assessing a combined hazard metric. This is shown by proposing a dynamic Conditional Value at Risk (CVaR) metric function.

**Contact:** Julian Kuehnert [(julian [dot] kuehnert [at] ibm [dot] com)](mailto:julian.kuehnert@ibm.com)


## Overview:

* ``datasets/``: Datasets in `.csv` format containing seasonal weather forecasts for rainfall and temperature and modeled malaria transmission dynamics.
* ``figures/``: Folder in which figures from results visualization are saved.
* ``models/``: Python scripts containing the R0 malaria model as well as the ML surrogate models (BLSTM and RFQR).
* ``trained_models/``: Folder which containes pre-trained ML models.
* ``utils/``: Python scripts for evaluation and visualization of results.
* ``malaria_modeling.ipynb``: Python notebook for simulating the climate-driven malaria transmission coefficient `R0`.
* ``ensemble_modeling.ipynb``: Python notebook for training surrogate models to predict uncertainties from weather forecast ensembles. 

## Datasets:
* Seasonal weather forecast ensembles of precipitation and temperature for point location in Nairobi, Kenya. Sample for 2021:

![precipitation](figures/precipitation_2021.png)
![temperature](figures/temperature_2021.png)

* Climate dependent malaria transmission coefficient R0, simulated based on the above climate forecast ensembles. Sample for 2021: 

![R0](figures/R0_2021.png)

## Running Benchmarks:

### Prerequisites and Python modules

Here we assume python3.9 and mac/linux, which are not necessarily mandatory.

    $ pip install jupyter numpy
    $ pip install -r requirements.txt

### Generate R0 for malaria model variables

Run through:

    Malaria_modeling.ipynb

for dataset preparation.  This will generate the ground truth data:

    datasets/forecasts_R0-malaria-model-variables.csv 
    
from

    datasets/forecasts_rain-temp_Nairobi_2017-2021.csv 

### Surrogate modeling

Surrogate ensemble modeling for dynamic climate impact models can be run as:

    ensemble_modeling.ipynb

Enjoy!

## Datasheet:
For transparency and accountability on the collection and content of the used datasets, a comprehensive datasheet on the dataset is provided. 


## License: 

See details under [LICENSE](LICENSE).

Dataset license:
    
    Dataset created by The Weather Company, an IBM business. This service is
    based on data and products of the European Center for Medium-range Weather
    Forecasts (ECMWF-Archive and ECMWF-RT). Generated using Copernicus Climate
    Change Service information [2019 and ongoing]. ECMWF Archive data published
    under a Creative Commons Attribution 4.0 International (CC BY 4.0) 
    https://creativecommons.org/licenses/by/4.0/ Disclaimer: Neither the 
    European Commission nor ECMWF is responsible for any use that may be made
    of the information it contains.
    
Code license:
    
    Apache License
    Version 2.0, January 2004
    http://www.apache.org/licenses/


## Citation:
    
