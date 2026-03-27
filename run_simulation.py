from dt_engine import DigitalTwinEngine
days = []
lai_list = []
stress_list = []
baseline_yield_list = []
corrected_yield_list = []

# Initialize DT
dt = DigitalTwinEngine(
    residual_model_path="models/dt_residual_model.pkl",
    initial_soil_moisture=120
)

# Example static inputs
soil = {
    "field_capacity_mm": 150,
    "wilting_point_mm": 60
}

crop = {
    "plant_density": 25,          # plants/m²
    "leaf_area_per_plant": 0.22   # m²
}

# Simulate 120 days
for day in range(120):
    dt.current_day = day + 1

    weather = {
        "rainfall_mm": 5,
        "et0_mm": 4,
        "t_max": 34,
        "t_min": 24,
        "radiation": 18
    }

    gdd_daily = 8

    dt.update_daily(weather, soil)
    seasonal_stress = dt.estimate_states(crop, gdd_daily)
    dt.adapt_parameters()
    output = dt.predict_yield(seasonal_stress)
    
    # Apply control actions
    dt.apply_controls(
        recommendations=output["recommendations"],
        soil=soil
    )
    days.append(output["day"])
    lai_list.append(output["LAI"])
    stress_list.append(output["seasonal_stress"])
    baseline_yield_list.append(output["baseline_yield"])
    corrected_yield_list.append(output["corrected_yield"])

    print(output)

import matplotlib.pyplot as plt

# -----------------------------
# Phase 4B: Diagnostic Plots
# -----------------------------

# 1. LAI evolution
plt.figure()
plt.plot(days, lai_list)
plt.xlabel("Day")
plt.ylabel("Leaf Area Index (LAI)")
plt.title("LAI Dynamics Over Season")
plt.grid(True)
plt.show()

# 2. Seasonal stress evolution
plt.figure()
plt.plot(days, stress_list)
plt.xlabel("Day")
plt.ylabel("Seasonal Stress")
plt.title("Seasonal Stress Accumulation")
plt.ylim(0.7, 1.0)
plt.grid(True)
plt.show()

# 3. Yield comparison
plt.figure()
plt.plot(days, baseline_yield_list, label="Baseline Yield")
plt.plot(days, corrected_yield_list, label="Corrected Yield")
plt.xlabel("Day")
plt.ylabel("Yield (kg/ha)")
plt.title("Baseline vs Corrected Yield")
plt.legend()
plt.grid(True)
plt.show()

