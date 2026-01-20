def calculate_avg_co2_reduction(df):
    return round(df["co2_reduction_percent"].mean(), 2)


def calculate_total_cost_savings(df):
    return round(df["cost_savings"].sum(), 2)


def calculate_total_packages(df):
    return int(df["quantity"].sum())
