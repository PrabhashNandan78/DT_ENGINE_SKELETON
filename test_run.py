# test_run.py
"""
Minimal smoke test for the DT engine skeleton.

What it checks (and nothing more):
  1. The physics model runs and returns a positive baseline yield.
  2. DigitalTwinEngine can be constructed and stepped through a few
     daily updates, producing a yield prediction by the end.

The engine normally loads a trained residual model from
models/dt_residual_model.pkl (produced by the DIGITALTWIN_ML notebooks).
That file is not part of this repository, so this test trains a tiny
throwaway LinearRegression on synthetic rows and saves it with joblib
into a temporary directory purely so the engine has a model to load.
No result from that stand-in model is meaningful — it only proves the
engine loop runs end to end.

Run:  python test_run.py
Requires: numpy, pandas, scikit-learn, joblib
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from physics_model import run_physics
from dt_engine import DigitalTwinEngine

FEATURES = ["seasonal_stress", "LAI_max", "leaf_growth_rate", "RUE",
            "harvest_index"]


def make_stand_in_residual_model(path):
    """Fit a trivial model with the feature layout the engine expects."""
    rows = [
        {"seasonal_stress": 0.80, "LAI_max": 5.0, "leaf_growth_rate": 0.07,
         "RUE": 1.4, "harvest_index": 0.38},
        {"seasonal_stress": 0.90, "LAI_max": 5.5, "leaf_growth_rate": 0.08,
         "RUE": 1.5, "harvest_index": 0.40},
        {"seasonal_stress": 0.75, "LAI_max": 4.8, "leaf_growth_rate": 0.06,
         "RUE": 1.3, "harvest_index": 0.36},
        {"seasonal_stress": 0.95, "LAI_max": 5.2, "leaf_growth_rate": 0.09,
         "RUE": 1.45, "harvest_index": 0.39},
    ]
    X = pd.DataFrame(rows, columns=FEATURES)
    y = [10.0, -5.0, 20.0, 0.0]
    model = LinearRegression().fit(X, y)
    joblib.dump(model, path)


def main():
    # 1. Physics model alone
    physics_out = run_physics({
        "days": 10,
        "seasonal_stress": 0.9,
        "LAI_max": 5.0,
        "leaf_growth_rate": 0.07,
        "RUE": 1.4,
        "harvest_index": 0.38,
    })
    assert physics_out["baseline_yield"] > 0, physics_out
    assert physics_out["biomass"] > 0, physics_out
    print("physics_model OK:", {k: round(v, 3)
                                for k, v in physics_out.items()})

    # 2. Engine: construct and step a few days
    soil = {"field_capacity_mm": 150, "wilting_point_mm": 60}
    crop = {"plant_density": 25, "leaf_area_per_plant": 0.22}
    weather = {"rainfall_mm": 5, "et0_mm": 4, "t_max": 34, "t_min": 24,
               "radiation": 18}

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "dt_residual_model.pkl")
        make_stand_in_residual_model(model_path)

        dt = DigitalTwinEngine(
            residual_model_path=model_path,
            initial_soil_moisture=120,
        )

        output = None
        for _ in range(3):
            dt.update_daily(weather, soil)
            seasonal_stress = dt.estimate_states(crop, gdd_daily=8)
            dt.adapt_parameters()
            output = dt.predict_yield(seasonal_stress)
            dt.apply_controls(output["recommendations"], soil)

    assert dt.day == 3, dt.day
    assert output is not None and output["day"] == 3, output
    assert output["LAI"] > 0.3, output  # LAI starts at 0.3 and must grow
    assert "plant_health" in output and "recommendations" in output, output
    print("DigitalTwinEngine OK after 3 days:", output)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
