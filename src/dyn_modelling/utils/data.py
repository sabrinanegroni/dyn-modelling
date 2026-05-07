""" Data utilities for cell lattice model and other. """

import numpy as np

def add_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to the data."""
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise