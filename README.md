# uplift-memof403

## Benchmark of Python packages for uplift modeling.

## Datasets

The datasets used for this benchmark are: 
* [Hillstrom's MineThatData E-Mail Analytics And Data Mining Challenge dataset](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html). The target variable is *visit*, we drop *conversion* and *spend*.
* [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). The target variable is *visit*, we drop *conversion* and *exposure*.

## Packages

The packages reviewed in this benchmark are:
* [CausalML](https://pypi.org/project/causalml/)
* [pyuplift](https://pypi.org/project/pyuplift/)
* [scikit-uplift](https://pypi.org/project/scikit-uplift/)

### CausalML

The models used are:
* S-Learner
* T-Learner
* X-Learner
* R-Learner
* Uplift Random Forest with Contextual Treatment Selection

The Meta-Learners are used with a Random Forest classifier.

### pyuplift

The models used all rely on the Class Transformation method. We test the following approaches with a Random Forest classifier:
* Lai
* Kane
* Jaskowski
* Pessimistic
* Reflective

### scikit-uplift
