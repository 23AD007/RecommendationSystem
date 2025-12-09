# src/etl/feature_engineering.py

from pathlib import Path
import pandas as pd

from src.ecopackdb.db_connect import get_engine


def load_cleaned() -> pd.DataFrame:
    path = Path("data") / "processed" / "cleaned_options.csv"
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} cleaned rows from {path.resolve()}")
    return df


def normalize_series(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - co2_impact_index
      - cost_efficiency_index
      - material_suitability_score
      - carbon_cost_ratio
      - region_pressure_score
      - material_complexity
    """

    # --- 1. CO2 Impact Index --- #
    co2 = df.get("region_adjusted_co2", df.get("base_co2"))
    rec = df.get("recyclability_percent", pd.Series(50, index=df.index))
    bio = df.get("biodegradability_score", pd.Series(5, index=df.index))

    co2_norm = normalize_series(co2) if co2 is not None else pd.Series(0.5, index=df.index)
    rec_norm = rec / 100.0
    bio_norm = bio / 10.0

    df["co2_impact_index"] = (
        0.5 * (1 - co2_norm) +
        0.3 * rec_norm +
        0.2 * bio_norm
    )

    # --- 2. Cost Efficiency Index --- #
    sust = df.get("base_sustainability", pd.Series(50, index=df.index))
    cost = df.get("region_adjusted_cost", df.get("base_cost"))

    cei_raw = sust / (cost + 1e-9)
    df["cost_efficiency_index"] = normalize_series(cei_raw)

    # --- 3. Material Suitability Score --- #
    risk = df.get("risk_of_damage_score", pd.Series(5, index=df.index))
    avail = df.get("availability_score", pd.Series(5, index=df.index))
    eco_pref = df.get("eco_preference_score", pd.Series(5, index=df.index))

    risk_norm = normalize_series(risk)
    avail_norm = avail / 10.0
    eco_pref_norm = eco_pref / 10.0
    sust_norm = sust / 100.0

    df["material_suitability_score"] = (
        0.4 * sust_norm +
        0.2 * avail_norm +
        0.2 * eco_pref_norm +
        0.2 * (1 - risk_norm)
    )

    # --- 4. Carbon-to-Cost Ratio --- #
    if "region_adjusted_co2" in df.columns and "region_adjusted_cost" in df.columns:
        df["carbon_cost_ratio"] = df["region_adjusted_co2"] / (df["region_adjusted_cost"] + 1e-9)

    # --- 5. Region Sustainability Pressure Score --- #
    rec_infra = df.get("recycling_infra_score", pd.Series(5, index=df.index))
    waste_mgmt = df.get("waste_mgmt_score", pd.Series(5, index=df.index))
    grid_co2 = df.get("grid_co2_kg_per_kwh", pd.Series(0.8, index=df.index))

    rec_infra_norm = rec_infra / 10.0
    waste_mgmt_norm = waste_mgmt / 10.0
    grid_norm = normalize_series(grid_co2)

    df["region_pressure_score"] = (
        0.4 * rec_infra_norm +
        0.3 * waste_mgmt_norm +
        0.3 * (1 - grid_norm)
    )

    # --- 6. Material Complexity Factor --- #
    strength = df.get("strength_kg", pd.Series(10, index=df.index))
    weight = df.get("weight_g_per_m2", pd.Series(50, index=df.index))
    barrier_o2 = df.get("barrier_oxygen_score", pd.Series(5, index=df.index))
    temp_max = df.get("temperature_max_c", pd.Series(40, index=df.index))

    strength_n = normalize_series(strength)
    weight_n = normalize_series(weight)
    barrier_n = barrier_o2 / 10.0
    temp_n = normalize_series(temp_max)

    df["material_complexity"] = (
        0.3 * strength_n +
        0.2 * weight_n +
        0.3 * barrier_n +
        0.2 * temp_n
    )

    print("\n=== Engineered feature summary ===")
    print(df[[
        "co2_impact_index",
        "cost_efficiency_index",
        "material_suitability_score",
        "carbon_cost_ratio",
        "region_pressure_score",
        "material_complexity",
    ]].describe())

    return df


def save_engineered(df: pd.DataFrame):
    out_dir = Path("data") / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "engineered_options.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved engineered data to CSV: {csv_path.resolve()}")

    # also push back to Postgres
    engine = get_engine()
    table_name = "engineered_product_material_options"
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Saved engineered data to Postgres table: {table_name}")


def main():
    df_clean = load_cleaned()
    df_feat = add_engineered_features(df_clean)
    save_engineered(df_feat)
    print("\nSample rows with engineered features:")
    print(df_feat.head())


if __name__ == "__main__":
    main()
