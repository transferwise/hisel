from . import kernels, select, hsic  # NOQA
try:
    import torch
    from . import torchkernels
except (ImportError, ModuleNotFoundError):
    pass
