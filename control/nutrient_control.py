def apply_nitrogen(leaf_growth_rate, remaining_budget):
    if remaining_budget <= 0:
        return leaf_growth_rate, remaining_budget

    increment = 0.01
    cost = 5.0

    leaf_growth_rate += increment
    remaining_budget -= cost

    return leaf_growth_rate, remaining_budget
