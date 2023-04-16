from . import kernels, select  # NOQA
try:
    import torch
    from . import torchkernels
except (ImportError, ModuleNotFoundError):
    pass
