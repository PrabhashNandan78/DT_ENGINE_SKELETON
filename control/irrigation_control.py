def apply_irrigation(soil_moisture, amount_mm=10, max_capacity=150):
    """
    Apply irrigation to soil moisture with physical limits.
    """
    new_moisture = soil_moisture + amount_mm
    return min(new_moisture, max_capacity)
