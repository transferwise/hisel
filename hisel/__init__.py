from . import kernels  # NOQA
try:
    import torch
    from . import torchkernels
except (ImportError, ModuleNotFoundError):
    pass
