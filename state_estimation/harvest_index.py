# state_estimation/harvest_index.py
import numpy as np


def estimate_harvest_index(
    biomass_series,
    yield_observed=None,
    min_hi=0.30,
    max_hi=0.45
):
    """
    Estimate Harvest Index (HI).

    Two modes:
    1) If observed yield is available (preferred)
    2) If not, return calibrated default (physiology-based)

    Parameters
    ----------
    biomass_series : list or np.ndarray
        Above-ground biomass values over season
    yield_observed : float or None
        Observed grain/pod yield at harvest (kg/ha or relative)
    min_hi : float
        Lower physiological bound
    max_hi : float
        Upper physiological bound

    Returns
    -------
    hi_est : float
        Estimated harvest index
    """

    biomass_series = np.array(biomass_series)

    # Use final biomass
    final_biomass = biomass_series[-1]

    if yield_observed is not None and final_biomass > 0:
        hi_est = yield_observed / final_biomass
    else:
        # fallback: typical groundnut HI
        hi_est = 0.38

    hi_est = float(np.clip(hi_est, min_hi, max_hi))
    return hi_est
