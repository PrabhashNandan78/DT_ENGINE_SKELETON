import matplotlib.pyplot as plt
from dt_engine import DigitalTwinEngine

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
