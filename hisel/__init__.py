from . import kernels, select, hsic  # NOQA
try:
    import torch   # NOQA
    from . import torchkernels  # NOQA
except (ImportError, ModuleNotFoundError):
    pass
