"""
Deterministic seed setter.

Section 4.1.4 of the thesis specifies that every experimental
configuration is repeated ten times with different random seeds, and that
the seed controls dataset shuffling, model parameter initialisation, data
augmentation sampling and batch ordering. To make the runs reproducible,
this module fixes the Python, NumPy and PyTorch seeds and disables the
non-deterministic cuDNN paths.

The ten seed values used in the thesis are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
They are pre-declared and must not be changed after the experiments
begin.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    Fix all random seeds and enable deterministic cuDNN behaviour.

    Args:
        seed: integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Deterministic reductions in cuBLAS require this environment
    # variable to be set before the first cuBLAS call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_seeds():
    """
    Return the ten pre-declared seeds used in the thesis.

    Returns:
        List of ten integers.
    """
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]