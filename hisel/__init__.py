from . import kernels, select, permutohedron, hsic, categorical,  feature_selection  # NOQA
try:
    import torch   # NOQA
    from . import torchkernels  # NOQA
except (ImportError, ModuleNotFoundError):
    pass
