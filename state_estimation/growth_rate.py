# state_estimation/growth_rate.py
import numpy as np


def estimate_leaf_growth_rate(
    lai_series,
    lai_max,
    min_r=0.02,
    max_r=0.15
):
    """
    Estimate leaf growth rate (r) from LAI time series
    using inverted logistic growth equation.

    Parameters
    ----------
    lai_series : list or np.ndarray
        LAI values over time (length >= 3)
    lai_max : float
        Estimated maximum LAI
    min_r : float
        Lower physiological bound
    max_r : float
        Upper physiological bound

    Returns
    -------
    r_est : float
        Estimated leaf growth rate
    """

    lai_series = np.array(lai_series)

    r_values = []

    for t in range(len(lai_series) - 1):
        lai_t = lai_series[t]
        lai_next = lai_series[t + 1]

        # Skip invalid or saturated points
        if lai_t <= 0.05 or lai_t >= 0.95 * lai_max:
            continue

        denom = lai_t * (1 - lai_t / lai_max)
        if denom <= 0:
            continue

        r_t = (lai_next - lai_t) / denom
        r_values.append(r_t)

    if len(r_values) == 0:
        # fallback to literature mean if insufficient data
        return 0.07

    r_est = np.median(r_values)
    r_est = float(np.clip(r_est, min_r, max_r))

    return r_est
