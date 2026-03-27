# state_estimation/lai.py
import numpy as np


def update_lai(
    lai_prev,
    gdd_daily,
    lai_max,
    gdd_half_max=600,
    stress_factor=1.0
):
    """
    Update daily LAI using thermal time (GDD-based growth).

    Parameters
    ----------
    lai_prev : float
        LAI from previous day
    gdd_daily : float
        Growing Degree Days for the day
    lai_max : float
        Maximum achievable LAI
    gdd_half_max : float
        GDD at which LAI reaches ~50% of LAI_max
    stress_factor : float
        Stress modifier (0–1), derived from seasonal stress

    Returns
    -------
    lai_new : float
        Updated LAI
    """

    # Logistic LAI growth driven by thermal time
    growth_rate = gdd_daily / (gdd_half_max + gdd_daily)

    lai_new = lai_prev + growth_rate * (lai_max - lai_prev) * stress_factor
    lai_new = min(lai_new, lai_max)

    return lai_new


def estimate_lai_max(
    plant_density,
    leaf_area_per_plant,
    stress_factor=1.0
):
    """
    Estimate LAI_max from field parameters.

    Parameters
    ----------
    plant_density : float
        Plants per m²
    leaf_area_per_plant : float
        Max leaf area per plant (m²)
    stress_factor : float
        Long-term stress modifier (0–1)

    Returns
    -------
    lai_max : float
        Estimated maximum LAI
    """

    lai_max = plant_density * leaf_area_per_plant * stress_factor
    return lai_max
