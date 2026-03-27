# physics_model.py
import numpy as np


def run_physics(state):
    """
    Physics-based crop growth and yield model for groundnut.

    Inputs (state dict):
        - days : int
        - seasonal_stress : float (0.7–1.0, mapped in DT engine)
        - LAI_max : float
        - leaf_growth_rate : float
        - RUE : float
        - harvest_index : float

    Outputs:
        - baseline_yield : float (kg/ha, relative scale)
        - biomass : float
        - final_LAI : float
    """

    days = state["days"]
    stress = state["seasonal_stress"]
    LAI_max = state["LAI_max"]
    r = state["leaf_growth_rate"]
    RUE = state["RUE"]
    HI = state["harvest_index"]

    # Initial conditions
    LAI = 0.3
    biomass = 0.0

    for _ in range(days):
        # Mean daily PAR (MJ/m²/day)
        PAR = 18.0

        # Logistic LAI growth with stress modulation
        LAI = LAI + r * LAI * (1 - LAI / LAI_max) * stress
        LAI = min(LAI, LAI_max)

        # Fraction of intercepted radiation
        IPAR = PAR * (1 - np.exp(-0.6 * LAI))

        # Biomass accumulation
        dB = RUE * IPAR * stress
        biomass += dB

    # Yield formation
    baseline_yield = biomass * HI

    return {
        "baseline_yield": baseline_yield,
        "biomass": biomass,
        "final_LAI": LAI
    }
