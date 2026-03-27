# state_estimation/stress.py
import numpy as np


def compute_raw_seasonal_stress(
    rainfall_mm,
    et0_mm,
    soil_moisture_mm,
    field_capacity_mm,
    wilting_point_mm,
    t_max,
    t_min,
    t_opt=(25.0, 30.0),
):
    """
    Computes RAW seasonal stress (0–1) from water + thermal stress.

    This function represents physiological crop stress derived from
    sensor / weather / soil data.

    IMPORTANT:
    - Output is 0–1 (biological reality)
    - Scaling to model range (0.7–1.0) is done later in DT engine

    Parameters
    ----------
    rainfall_mm : float
        Daily rainfall (mm)
    et0_mm : float
        Daily reference evapotranspiration (mm)
    soil_moisture_mm : float
        Current soil moisture (mm)
    field_capacity_mm : float
        Soil field capacity (mm)
    wilting_point_mm : float
        Soil wilting point (mm)
    t_max : float
        Daily maximum temperature (°C)
    t_min : float
        Daily minimum temperature (°C)
    t_opt : tuple(float, float)
        Optimal temperature range for groundnut (°C)

    Returns
    -------
    raw_stress : float
        Combined water × thermal stress in [0, 1]
    updated_soil_moisture : float
        Updated soil moisture after water balance
    """

    # -------------------------
    # WATER STRESS COMPONENT
    # -------------------------
    updated_soil_moisture = soil_moisture_mm + rainfall_mm - et0_mm

    # Constrain soil moisture within physical limits
    updated_soil_moisture = np.clip(
        updated_soil_moisture,
        wilting_point_mm,
        field_capacity_mm
    )

    if updated_soil_moisture <= wilting_point_mm:
        water_stress = 0.0
    elif updated_soil_moisture >= field_capacity_mm:
        water_stress = 1.0
    else:
        water_stress = (
            (updated_soil_moisture - wilting_point_mm)
            / (field_capacity_mm - wilting_point_mm)
        )

    # -------------------------
    # THERMAL STRESS COMPONENT
    # -------------------------
    t_mean = (t_max + t_min) / 2.0
    t_opt_mid = np.mean(t_opt)

    if t_opt[0] <= t_mean <= t_opt[1]:
        thermal_stress = 1.0
    else:
        thermal_stress = max(
            0.0,
            1.0 - abs(t_mean - t_opt_mid) / 15.0
        )

    # -------------------------
    # COMBINED RAW STRESS
    # -------------------------
    raw_stress = water_stress * thermal_stress
    raw_stress = float(np.clip(raw_stress, 0.0, 1.0))

    return raw_stress, updated_soil_moisture
