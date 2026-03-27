# plant_health/health_logic.py
import numpy as np


def compute_plant_health(
    day,
    seasonal_stress,
    lai_history,
    baseline_yield,
    corrected_yield
):
    """
    Compute plant health score and status from Digital Twin states.
    """

    # -------------------------
    # 1. Stress-based health
    # seasonal_stress in [0.7, 1.0]
    # -------------------------
    stress_health = (seasonal_stress - 0.7) / 0.3
    stress_health = np.clip(stress_health, 0.0, 1.0)

    # -------------------------
    # 2. Growth-based health (LAI trend)
    # -------------------------
    
    if lai_history[-1] >= 3.0:
        growth_health = 1.0   # canopy established
    elif len(lai_history) >= 5:
        lai_trend = lai_history[-1] - lai_history[-5]
        growth_health = np.clip(lai_trend / 0.3, 0.0, 1.0)
    else:
        growth_health = 1.0


    # -------------------------
    # 2b. Maturity-aware correction
    # -------------------------
    lai_current = lai_history[-1]
    lai_max_seen = max(lai_history)

    if lai_current > 0.9 * lai_max_seen:
        # slowing growth near canopy closure is normal
        growth_health = np.clip(growth_health, 0.0, 1.0)



    # -------------------------
    # 3. Yield consistency health
    # -------------------------
    if day < 40:
        yield_health = 1.0
    else:
        deviation = abs(baseline_yield - corrected_yield) / max(baseline_yield, 1.0)
        yield_health = np.clip(1.0 - deviation, 0.0, 1.0)

    # -------------------------
    # Final health score (weighted)
    # -------------------------
    health_score = (
        0.4 * stress_health +
        0.4 * growth_health +
        0.2 * yield_health
    )

    # -------------------------
    # Health classification
    # -------------------------
    if health_score >= 0.75:
        status = "Healthy"
    elif health_score >= 0.5:
        status = "Moderate Stress"
    else:
        status = "Severe Stress"

    return {
        "health_score": round(float(health_score), 3),
        "status": status,
        "stress_health": round(float(stress_health), 3),
        "growth_health": round(float(growth_health), 3),
        "yield_health": round(float(yield_health), 3)
    }
