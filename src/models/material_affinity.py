import pandas as pd

def learn_material_affinity(df: pd.DataFrame):
    """
    Learn product_category → material affinity multipliers from data.
    Returns: dict[product_category][material] -> multiplier
    """

    # We assume:
    # - df["product_category"]
    # - df["material"]
    # - df["recommended"] ∈ {0,1} or relevance score

    # Mean recommendation rate per (category, material)
    grp = (
        df
        .groupby(["product_category", "material"])["recommended"]
        .mean()
        .reset_index()
    )

    # Global baseline per material
    material_baseline = (
        df
        .groupby("material")["recommended"]
        .mean()
        .to_dict()
    )

    affinity = {}

    for _, row in grp.iterrows():
        cat = row["product_category"]
        mat = row["material"]
        rate = row["recommended"]

        baseline = material_baseline.get(mat, 0.5)

        # Ratio → multiplier
        multiplier = rate / max(baseline, 1e-6)

        # Clamp to stay sane
        multiplier = max(0.8, min(1.2, multiplier))

        affinity.setdefault(cat, {})[mat] = round(multiplier, 3)

    return affinity
