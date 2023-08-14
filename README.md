# hisel
Feature selection tool based on Hilbert-Schmidt Independence Criterion

This package provides an implementtion of the HSIC Lasso of [Yamada, M. et al. (2012)](https://arxiv.org/abs/1202.0515). 

Usage is demontrated in the notebooks and in the scripts available under `examples/`. 


## Installation

The package `hisel` is available from `arti`. You can install it via `pip`. 
While on the Wise-VPN, in the environment where you intende to sue `hisel`, just do
```
pip install hisel --index-url=https://arti.tw.ee/artifactory/api/pypi/pypi-virtual/simple
```


## Why is this cool?

Examples of where `hisel` outperforms the methods in 
[sklearn.feature\_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)
are given in the notebooks
`ensemble-example.ipynb`
and
`nonlinear-trasnform.ipynb`.
