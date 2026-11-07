# DT_ENGINE_SKELETON — Early Digital Twin Engine Prototype

An early snapshot of the simulation engine built for **DigitalTwin-Agri**,
the research behind the IEEE paper *"DigitalTwin-Agri: A Hybrid Digital
Twin Framework for Next-Generation Smart Agriculture"*.

This repository is kept as the record of that first engine. It has been
**superseded by [TWIN](https://github.com/PrabhashNandan78/TWIN)**, which
re-implements the same ideas as a plant-level farm / plant / sensor twin
package — new work should start there. The ML side of the project (which
also trains the residual model this engine loads) lives in
[DIGITALTWIN_ML](https://github.com/PrabhashNandan78/DIGITALTWIN_ML).

## What the engine does

`DigitalTwinEngine` (`dt_engine.py`) tracks one groundnut crop season day
by day:

1. **Daily update** — a simple soil water balance (rainfall − ET₀, clipped
   between wilting point and field capacity) plus temperature gives a raw
   seasonal stress value (`state_estimation/stress.py`).
2. **State estimation** — LAI is advanced with a GDD-driven logistic update,
   and LAI-max / leaf growth rate are estimated from the crop parameters
   and the LAI history (`state_estimation/lai.py`, `growth_rate.py`;
   `rue.py` and `harvest_index.py` hold the corresponding estimators for
   radiation-use efficiency and harvest index).
3. **Yield prediction** — `physics_model.run_physics()` integrates a
   logistic-LAI, radiation-interception biomass model to a baseline yield;
   from day 40 on, a saved ML residual model (joblib) corrects that
   baseline — the hybrid physics + ML pattern used across this project.
4. **Health & decisions** — `plant_health/health_logic.py` scores plant
   health from stress, LAI trend and yield consistency, and
   `decision_support/recommendations.py` turns that into actions such as
   "Increase irrigation" or "Apply nitrogen fertilizer".
5. **Control** — `control/` applies those actions back onto the engine
   state (irrigation raises soil moisture; nitrogen raises the leaf growth
   rate, guarded by crop stage, canopy closure and a nitrogen budget).

## Phase structure

| Path | Phase / role |
|---|---|
| `dt_engine.py`, `physics_model.py`, `state_estimation/`, `plant_health/`, `decision_support/`, `control/` | The core engine loop described above. |
| `run_simulation.py` | The main driver: a 120-day season with fixed example weather, printing each day's output and plotting LAI, stress and yield. |
| `phase_4B_diagnostics/` | Diagnostic plots (LAI dynamics, stress accumulation, baseline vs corrected yield). The script is a plotting fragment that works on the series built by `run_simulation.py`. |
| `phase_4C_4D_uncertainty/` | Monte Carlo uncertainty analysis: 200 randomised 120-day runs, producing the yield-uncertainty histogram and plant-health risk distribution committed under `outputs/`. |

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn joblib
python test_run.py        # smoke test — see below
python run_simulation.py  # full 120-day example season
```

> **Note:** both the engine and `run_simulation.py` load a trained
> residual model from `models/dt_residual_model.pkl`. That file is not
> committed here — it is produced by notebook
> `6digital_twin_feedback.ipynb` in
> [DIGITALTWIN_ML](https://github.com/PrabhashNandan78/DIGITALTWIN_ML).
> `test_run.py` does not need it: it trains a throwaway stand-in model in
> a temporary directory so the engine loop can be exercised end to end.

## Smoke test

`test_run.py` is a minimal, self-contained check: it verifies the physics
model returns a positive baseline yield, builds `DigitalTwinEngine` with
the stand-in residual model, steps it through a few days of example
weather, and asserts the season state advances and a yield prediction is
produced. Run it with `python test_run.py`.

## The DigitalTwin-Agri family

- **[DT_ENGINE_SKELETON](https://github.com/PrabhashNandan78/DT_ENGINE_SKELETON)**
  (this repo) — the early engine prototype / snapshot.
- **[TWIN](https://github.com/PrabhashNandan78/TWIN)** — the later engine
  that supersedes it.
- **[DIGITALTWIN_ML](https://github.com/PrabhashNandan78/DIGITALTWIN_ML)**
  — the ML notebooks and datasets behind the paper.
- **[smartfarming](https://github.com/PrabhashNandan78/smartfarming)** — the
  original combined upload of all of the above.
