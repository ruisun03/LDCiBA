"""Coefficient support and out-of-sample predictive evidence (Section 3.3)."""

import math

import numpy as np
from scipy.special import ndtr


def coefficient_support(full_mean, variance, threshold):
    """Eq. (7). full_mean is the reference coefficient plus the posterior increment."""
    if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(variance) or variance <= 0:
        raise ValueError("The magnitude threshold and posterior variance must be finite and positive")
    sd = np.sqrt(variance)
    inside = ndtr((threshold - full_mean) / sd) - ndtr((-threshold - full_mean) / sd)
    return float(1.0 - inside)


def gaussian_log_density(observation, prediction, variance):
    """One target's log predictive density, using the mean and variance issued before adaptation."""
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Predictive variance must be finite and positive")
    return -0.5 * (math.log(2 * math.pi * variance) + (observation - prediction) ** 2 / variance)


def validation_log_evidence(candidate_log_density, accepted_log_density):
    """Eq. (8). Match one pre-update log density per model and arrival in the subsequent validation window."""
    candidate = np.asarray(candidate_log_density, dtype=np.float64)
    accepted = np.asarray(accepted_log_density, dtype=np.float64)
    if candidate.ndim != 1 or candidate.shape != accepted.shape:
        raise ValueError("Both models must supply log densities for the same observations")
    return float(np.sum(candidate - accepted))
