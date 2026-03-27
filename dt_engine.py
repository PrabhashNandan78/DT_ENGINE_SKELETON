# dt_engine.py
import joblib
from physics_model import run_physics

from state_estimation.stress import compute_raw_seasonal_stress
from state_estimation.lai import update_lai, estimate_lai_max
from state_estimation.growth_rate import estimate_leaf_growth_rate

from plant_health.health_logic import compute_plant_health
from decision_support.recommendations import generate_recommendations
from control.irrigation_control import apply_irrigation
from control.nutrient_control import apply_nitrogen


class DigitalTwinEngine:
    def __init__(self, residual_model_path, initial_soil_moisture):
        # ML model
        self.residual_model = joblib.load(residual_model_path)

        # --- STATE VARIABLES ---
        self.day = 0
        self.soil_moisture = initial_soil_moisture
        self.nitrogen_budget = 100      # kg/ha
        self.irrigation_budget = 300   # mm
        self.last_nitrogen_day = -999


        self.raw_stress_history = []
        self.lai_history = []
        self.biomass_history = []
        self.radiation_history = []

        # --- PARAMETERS ---
        self.LAI = 0.3
        self.LAI_max = None
        self.leaf_growth_rate = None
        self.RUE = 1.4
        self.harvest_index = 0.38

        # Health state
        self.plant_health = None

    def get_crop_stage(self):
        if self.day < 25:
            return "early_vegetative"
        elif self.day < 50:
            return "late_vegetative"
        elif self.day < 70:
            return "flowering"
        else:
            return "pod_filling"

    
    def apply_controls(self, recommendations, soil):
        """
        Modify DT internal state based on decision support.
        """

        if "Increase irrigation" in recommendations:
            self.soil_moisture = apply_irrigation(
                soil_moisture=self.soil_moisture,
                amount_mm=12,
                max_capacity=soil["field_capacity_mm"]
            )

        stage = self.get_crop_stage()

        # Nitrogen allowed ONLY in vegetative stages
        if stage not in ["early_vegetative", "late_vegetative"]:
            return

        # Do not apply nitrogen after canopy closure
        if self.LAI >= 3.5:
            return
        if self.day - self.last_nitrogen_day < 10:
            return
        # No nitrogen after flowering / pod filling
        if self.day > 60:
            return
        

        if "Apply nitrogen fertilizer" in recommendations:
            if self.leaf_growth_rate is not None:
                self.leaf_growth_rate, self.nitrogen_budget = apply_nitrogen(
                    self.leaf_growth_rate,
                    self.nitrogen_budget
        )
                self.last_nitrogen_day = self.day



    # --------------------------------------------------
    # DAILY UPDATE
    # --------------------------------------------------
    def update_daily(self, weather, soil):
        raw_stress, self.soil_moisture = compute_raw_seasonal_stress(
            rainfall_mm=weather["rainfall_mm"],
            et0_mm=weather["et0_mm"],
            soil_moisture_mm=self.soil_moisture,
            field_capacity_mm=soil["field_capacity_mm"],
            wilting_point_mm=soil["wilting_point_mm"],
            t_max=weather["t_max"],
            t_min=weather["t_min"],
        )

        self.raw_stress_history.append(raw_stress)
        self.radiation_history.append(weather["radiation"])
        self.day += 1

    # --------------------------------------------------
    # STATE ESTIMATION
    # --------------------------------------------------
    def estimate_states(self, crop, gdd_daily):

        if self.LAI_max is None:
            self.LAI_max = estimate_lai_max(
                plant_density=crop["plant_density"],
                leaf_area_per_plant=crop["leaf_area_per_plant"],
                stress_factor=1.0
            )

        raw_seasonal_stress = sum(self.raw_stress_history) / self.day
        seasonal_stress = 0.7 + 0.3 * raw_seasonal_stress

        self.LAI = update_lai(
            lai_prev=self.LAI,
            gdd_daily=gdd_daily,
            lai_max=self.LAI_max,
            stress_factor=seasonal_stress
        )

        self.lai_history.append(self.LAI)
        return seasonal_stress

    # --------------------------------------------------
    # PARAMETER ADAPTATION
    # --------------------------------------------------
    def adapt_parameters(self):
        if self.leaf_growth_rate is None and len(self.lai_history) >= 7:
            self.leaf_growth_rate = estimate_leaf_growth_rate(
                self.lai_history,
                self.LAI_max
            )

    # --------------------------------------------------
    # YIELD PREDICTION + HEALTH + DECISION
    # --------------------------------------------------
    def predict_yield(self, seasonal_stress):

        state = {
            "days": self.day,
            "seasonal_stress": seasonal_stress,
            "LAI_max": self.LAI_max,
            "leaf_growth_rate": self.leaf_growth_rate or 0.07,
            "RUE": self.RUE,
            "harvest_index": self.harvest_index
        }

        physics_out = run_physics(state)
        self.biomass_history.append(physics_out["biomass"])

        import pandas as pd

        ml_features = pd.DataFrame([{
            "seasonal_stress": state["seasonal_stress"],
            "LAI_max": state["LAI_max"],
            "leaf_growth_rate": state["leaf_growth_rate"],
            "RUE": state["RUE"],
            "harvest_index": state["harvest_index"]
        }])

        residual = self.residual_model.predict(ml_features)[0]


        if self.day < 40:
            corrected_yield = physics_out["baseline_yield"]
        else:
            corrected_yield = physics_out["baseline_yield"] + residual

        # --- Plant health ---
        self.plant_health = compute_plant_health(
            day=self.day,
            seasonal_stress=seasonal_stress,
            lai_history=self.lai_history,
            baseline_yield=physics_out["baseline_yield"],
            corrected_yield=corrected_yield
        )
    
        # --- Decision support ---
        stage = self.get_crop_stage()

        recommendations = generate_recommendations(
            self.plant_health,
            crop_stage=stage,
            lai=self.LAI,
            days_since_last_nitrogen=self.day - self.last_nitrogen_day
        )



        return {
            "day": self.day,
            "seasonal_stress": seasonal_stress,
            "LAI": self.LAI,
            "baseline_yield": physics_out["baseline_yield"],
            "corrected_yield": corrected_yield,
            "plant_health": self.plant_health,
            "recommendations": recommendations
        }
