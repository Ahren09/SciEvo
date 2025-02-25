import os
import os.path as osp
import random

import numpy as np


def check_cwd():
    basename = osp.basename(osp.normpath(os.getcwd()))
    assert basename.lower() in [
        "scievo"], "Please run this file from parent directory (SciEvo/)"


def project_setup():
    check_cwd()
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)


def set_seed(seed, use_torch=True):
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def get_GPU_memory_allocated_to_tensor(t):
    # calculate the GPU memory occupied by t
    memory_in_MB = t.element_size() * t.nelement() / 1024 / 1024
    print(f"Tensor occupies {memory_in_MB:.2f} MB of GPU memory.")
    return memory_in_MB
