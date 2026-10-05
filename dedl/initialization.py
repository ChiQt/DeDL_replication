"""Shared initial arrays for comparable starting predictions."""
import numpy as np


def initial_weights(seed, dimension):
    rng = np.random.default_rng(seed)
    weights = []
    for output_dimension in [1, 3]:
        widths = [dimension, 20, 20, 20, output_dimension]
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            limit = np.sqrt(6 / (input_width + output_width))
            kernel = rng.uniform(-limit, limit, (input_width, output_width)).astype(np.float32)
            weights.append((kernel, np.zeros(output_width, dtype=np.float32)))
    return weights
