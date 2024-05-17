import random
import os

import torch
import numpy as np

import logging


LOGGER = logging.getLogger(name="infotropy.utils.random")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    """Set the random number generation seed globally for `torch`, `numpy` and `random`"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    LOGGER.info("Set 'numpy', 'random' and 'torch' random seed to %s", seed)


def get_random_seed():
    """Return a random seed between 0 and 2**32 - 1"""
    return random.randint(a=0, b=2**32 - 1)
