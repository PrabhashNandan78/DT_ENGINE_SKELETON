def generate_recommendations(
    plant_health,
    crop_stage,
    lai,
    days_since_last_nitrogen
):
    stress_h = plant_health["stress_health"]
    growth_h = plant_health["growth_health"]
    yield_h = plant_health["yield_health"]

    actions = []

    # Irrigation
    if stress_h < 0.8:
        actions.append("Increase irrigation")

    # Nitrogen (guarded)
    if (
        growth_h < 0.7
        and crop_stage in ["early_vegetative", "late_vegetative"]
        and lai < 3.5
        and days_since_last_nitrogen >= 10
    ):
        actions.append("Apply nitrogen fertilizer")

    if yield_h < 0.85:
        actions.append("Inspect crop for pest or disease")

    if yield_h < 0.85 and stress_h > 0.9:
        actions.append("Investigate model mismatch")

    if not actions:
        actions.append("Crop conditions are optimal")

    return actions
