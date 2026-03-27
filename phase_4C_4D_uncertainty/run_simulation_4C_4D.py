import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings

# -------------------------------------------------
# Silence sklearn feature-name warnings (intentional)
# -------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# -------------------------------------------------
# Fix Python path (robust)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from dt_engine import DigitalTwinEngine

# -------------------------------------------------
# Experiment configuration
# -------------------------------------------------
N_RUNS = 200
SIM_DAYS = 120

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Static inputs
soil = {
    "field_capacity_mm": 150,
    "wilting_point_mm": 60
}

crop = {
    "plant_density": 25,
    "leaf_area_per_plant": 0.22
}

# Result containers
final_yields = np.zeros(N_RUNS)
final_health_scores = np.zeros(N_RUNS)

print(f"Starting Monte Carlo: {N_RUNS} runs × {SIM_DAYS} days")

# -------------------------------------------------
# Phase 4C + 4D: Monte Carlo loop
# -------------------------------------------------
for run in range(N_RUNS):

    if run % 10 == 0:
        print(f"Monte Carlo progress: {run}/{N_RUNS}")

    dt = DigitalTwinEngine(
        residual_model_path=os.path.join(BASE_DIR, "models", "dt_residual_model.pkl"),
        initial_soil_moisture=np.random.uniform(90, 140)
    )

    for _ in range(SIM_DAYS):
        weather = {
            "rainfall_mm": max(0, np.random.normal(5, 2)),
            "et0_mm": max(2, np.random.normal(4, 0.8)),
            "t_max": np.random.normal(34, 1.5),
            "t_min": np.random.normal(24, 1.2),
            "radiation": max(10, np.random.normal(18, 2))
        }


        gdd_daily = 8

        dt.update_daily(weather, soil)
        seasonal_stress = dt.estimate_states(crop, gdd_daily)
        dt.adapt_parameters()

        output = dt.predict_yield(seasonal_stress)

        dt.apply_controls(
            recommendations=output["recommendations"],
            soil=soil
        )

    # Store final season outcomes
    final_yields[run] = output["corrected_yield"]
    final_health_scores[run] = output["plant_health"]["health_score"]

print("Monte Carlo completed.")

# -------------------------------------------------
# Phase 4C: Yield uncertainty analysis
# -------------------------------------------------
p10, p50, p90 = np.percentile(final_yields, [10, 50, 90])

plt.figure(figsize=(8, 5))
yield_range = final_yields.max() - final_yields.min()

if yield_range < 1e-6:
    bins = 1
else:
    bins = min(25, int(np.sqrt(len(final_yields))))

plt.hist(final_yields, bins=bins, alpha=0.75)

plt.axvline(p10, linestyle="--", label="P10")
plt.axvline(p50, linestyle="-", label="P50 (Median)")
plt.axvline(p90, linestyle="--", label="P90")
plt.xlabel("Final Yield (kg/ha)")
plt.ylabel("Frequency")
plt.title("Yield Uncertainty Distribution (Phase 4C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yield_histogram.png"))
plt.close()

# -------------------------------------------------
# Phase 4D: Plant health risk analysis
# -------------------------------------------------
plt.figure(figsize=(8, 5))
health_range = final_health_scores.max() - final_health_scores.min()

if health_range < 1e-6:
    bins = 1
else:
    bins = min(20, int(np.sqrt(len(final_health_scores))))

plt.hist(final_health_scores, bins=bins, alpha=0.75)

plt.axvline(0.75, linestyle="--", color="green", label="Healthy threshold")
plt.axvline(0.50, linestyle="--", color="orange", label="Moderate stress threshold")
plt.xlabel("Final Plant Health Score")
plt.ylabel("Frequency")
plt.title("Plant Health Risk Distribution (Phase 4D)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "health_risk_curve.png"))
plt.close()

# -------------------------------------------------
# Faculty-ready console summary
# -------------------------------------------------
print("\n--- Phase 4C: Yield Uncertainty ---")
print(f"P10 Yield : {p10:.2f} kg/ha")
print(f"P50 Yield : {p50:.2f} kg/ha")
print(f"P90 Yield : {p90:.2f} kg/ha")

print("\n--- Phase 4D: Plant Health Risk ---")
print(f"Mean Health Score           : {final_health_scores.mean():.3f}")
print(f"Moderate Stress Probability : {(final_health_scores < 0.75).mean():.2%}")
print(f"Severe Stress Probability   : {(final_health_scores < 0.50).mean():.2%}")
