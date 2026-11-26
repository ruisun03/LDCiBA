"""Gaussian structural-increment operations from GaussianFilter (Sections 3.2–3.3)."""

import numpy as np


def random_walk_prior(covariance, process_variances):
    """Return P + Q for one arrival with diagonal process noise; the mean carries over."""
    prior = np.array(covariance, dtype=np.float64, copy=True)
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
        raise ValueError("The state covariance must be square")
    process_variances = np.broadcast_to(process_variances, (len(prior),))
    if not np.isfinite(process_variances).all() or np.any(process_variances < 0):
        raise ValueError("Process variances must be finite and nonnegative")
    prior.flat[::len(prior) + 1] += process_variances
    return prior


def predictive_moments(reference_prediction, features, increment_mean, prior_covariance, noise_variance):
    """One-step mean and S in Eq. (3). The reference prediction already includes B0."""
    features = np.asarray(features, dtype=np.float64)
    prediction = float(reference_prediction + features @ increment_mean)
    variance = float(noise_variance + features @ prior_covariance @ features)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Predictive variance must be finite and positive")
    return prediction, variance


def update_admitted(increment_mean, prior_covariance, features, innovation, effective_noise_variance):
    """Eq. (5) for one admitted observation; pass pre-update innovation and R = sigma² / w."""
    if not np.isfinite(effective_noise_variance) or effective_noise_variance <= 0:
        raise ValueError("Effective observation variance must be finite and positive")
    features = np.asarray(features, dtype=np.float64)
    projected = prior_covariance @ features
    gain = projected / (features @ projected + effective_noise_variance)
    mean = increment_mean + gain * innovation
    # Joseph form is algebraically equivalent to the covariance update in Eq. (5).
    transform = np.eye(len(features)) - np.outer(gain, features)
    covariance = transform @ prior_covariance @ transform.T + effective_noise_variance * np.outer(gain, gain)
    return mean, (covariance + covariance.T) * 0.5


def expand_posterior(mean, covariance, new_variance):
    """Append one missing relation after the current update, with the independent prior in Eq. (6)."""
    if not np.isfinite(new_variance) or new_variance <= 0:
        raise ValueError("A new relation needs a positive, finite prior variance")
    size = len(mean)
    expanded = np.zeros((size + 1, size + 1), dtype=np.float64)
    expanded[:size, :size] = covariance
    expanded[size, size] = new_variance
    return np.r_[mean, 0.0], expanded


def marginalize_posterior(mean, covariance, keep):
    """Retain the marginal on the supplied coordinate indices, including all reference coordinates."""
    keep = np.asarray(keep, dtype=np.intp)
    return mean[keep], covariance[np.ix_(keep, keep)]
