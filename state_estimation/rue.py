# state_estimation/rue.py
import numpy as np


def estimate_rue(
    biomass_series,
    lai_series,
    radiation_series,
    k_extinction=0.6,
    min_rue=0.8,
    max_rue=2.0
):
    """
    Estimate Radiation Use Efficiency (RUE) from time series data.
    """

    biomass_series = np.array(biomass_series)
    lai_series = np.array(lai_series)
    radiation_series = np.array(radiation_series)

    # Need at least 2 points
    if len(biomass_series) < 2:
        return 1.4  # fallback

    # ΔBiomass (length n-1)
    delta_biomass = np.diff(biomass_series)

    # Intercepted radiation for SAME intervals (length n-1)
    ipar_series = radiation_series[:-1] * (
        1 - np.exp(-k_extinction * lai_series[:-1])
    )

    # Valid points
    valid = (delta_biomass > 0) & (ipar_series > 0)

    if np.sum(valid) == 0:
        return 1.4

    rue_values = delta_biomass[valid] / ipar_series[valid]
    rue_est = np.median(rue_values)

    return float(np.clip(rue_est, min_rue, max_rue))
