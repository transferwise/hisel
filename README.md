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

### Install from source

#### Basic installation:
Checkout the repo and navigate to the root directory. Then, 

```
poetry install
```



#### Installation with GPU support via [CuPy](https://cupy.dev/):
You need to have cuda-toolkit installed and you need to know its version.
To know that, you can do 
```
nvidia-smi
```
and read the cuda version from the top right corner of the table that is printed out. 
Once you know your version of `cuda`, do 
```
poetry install -E cudaXXX
```
where `cudaXXX` is one of the following:
`cuda102` if you have version 10.2;
`cuda110` if you have version 11.0;
`cuda111` if you have version 11.1;
`cuda11x` if you have version 11.2 - 11.8;
`cuda12x` if you have version 12.x.
This aligns to the [installation guide of CuPy](https://docs.cupy.dev/en/stable/install.html#installing-cupy).



## Why is this cool?

Examples of where `hisel` outperforms the methods in 
[sklearn.feature\_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)
are given in the notebooks
`ensemble-example.ipynb`
and
`nonlinear-trasnform.ipynb`.
