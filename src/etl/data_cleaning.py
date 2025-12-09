# src/etl/data_cleaning.py

from pathlib import Path
import pandas as pd

from src.ecopackdb.db_connect import get_engine

VIEW_NAME = "v_product_material_options"   # view you already have


def load_raw() -> pd.DataFrame:
    engine = get_engine()
    query = f"SELECT * FROM {VIEW_NAME};"
    df = pd.read_sql(query, engine)
    print(f"\nLoaded {len(df)} rows from {VIEW_NAME}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== BEFORE CLEANING ===")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isna().sum())

    # 1) Standardize text columns
    if "material_type" in df.columns:
        df["material_type"] = (
            df["material_type"].astype(str)
            .str.strip()
            .str.lower()
            .str.title()
        )

    if "material_name" in df.columns:
        df["material_name"] = df["material_name"].astype(str).str.strip()

    # 2) Drop rows missing critical identifiers / target
    essential = [
        c for c in ["product_id", "material_id", "region_id", "final_score"]
        if c in df.columns
    ]
    if essential:
        df = df.dropna(subset=essential)

    # 3) Group-wise medians for material-specific behaviour
    group_median_cols = ["biodegradability_score", "recyclability_percent"]
    for col in group_median_cols:
        if col in df.columns and "material_type" in df.columns:
            df[col] = df.groupby("material_type")[col].transform(
                lambda s: s.fillna(s.median())
            )

    # 4) Global medians for generic numeric cols
    global_median_cols = [
        "base_cost",
        "base_co2",
        "region_adjusted_cost",
        "region_adjusted_co2",
        "availability_score",
        "risk_of_damage_score",
        "base_sustainability",
        "eco_preference_score",
        "price_sensitivity_score",
        "premium_preference_score",
    ]
    for col in global_median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 5) Remove impossible values
    if "base_cost" in df.columns:
        df = df[df["base_cost"] > 0]
    if "base_co2" in df.columns:
        df = df[df["base_co2"] >= 0]
    if "region_adjusted_cost" in df.columns:
        df = df[df["region_adjusted_cost"] > 0]
    if "region_adjusted_co2" in df.columns:
        df = df[df["region_adjusted_co2"] >= 0]

    # 6) Mild outlier clipping for cost and CO2
    def clip_col(col, q_low=0.01, q_high=0.99):
        if col in df.columns:
            low = df[col].quantile(q_low)
            high = df[col].quantile(q_high)
            df[col] = df[col].clip(lower=low, upper=high)

    clip_col("region_adjusted_cost")
    clip_col("region_adjusted_co2")

    print("\n=== AFTER CLEANING ===")
    print(df.isna().sum())
    print("\nSample rows after cleaning:")
    print(df.head())

    return df


def save_cleaned(df: pd.DataFrame):
    # CSV
    out_dir = Path("data") / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cleaned_options.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved cleaned data to CSV: {csv_path.resolve()}")

    # back into Postgres as a table
    engine = get_engine()
    table_name = "cleaned_product_material_options"
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Saved cleaned data to Postgres table: {table_name}")


def main():
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    save_cleaned(df_clean)


if __name__ == "__main__":
    main()
