"""Historical inputs ordered by lag, then source variable (Section 3.1)."""

import numpy as np


def lag_design(values, order):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= order:
        raise ValueError('Insufficient past observations')
    return np.concatenate([values[order-k:len(values)-k]
                           for k in range(1, order+1)], axis=1)
