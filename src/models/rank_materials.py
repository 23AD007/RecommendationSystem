import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.utils.preprocessing import load_and_prepare_data

def rank_materials():
    # Load and preprocess data
    X, y = load_and_prepare_data("data/processed/training_dataset.csv")

    # Train a quick model for ranking
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=100,  # Quick training
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # Load original data for ranking
    df = pd.read_csv("data/processed/training_dataset.csv")

    # Predict scores on full dataset
    scores = model.predict_proba(X)[:, 1]

    df["recommendation_score"] = scores

    ranked = df.sort_values("recommendation_score", ascending=False)

    print(ranked[[
        "product_category",
        "recommendation_score",
        "fragility_score",
        "sustainability_priority",
        "durability_requirement"
    ]].head(10))


if __name__ == "__main__":
    rank_materials()
