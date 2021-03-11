# uplift-memof403

## Benchmark of Python packages for uplift modeling.

## Datasets

The datasets used for this benchmark are: 
* [Hillstrom's MineThatData E-Mail Analytics And Data Mining Challenge dataset](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html). The target variable is *visit*, we drop *conversion* and *spend*.
* [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). The target variable is *visit*, we drop *conversion* and *exposure*.

## Packages

The packages reviewed in this benchmark are:
* [CausalML](https://pypi.org/project/causalml/)

### CausalML

The models used are:
* S-Learner
* T-Learner
* X-Learner
* R-Learner
