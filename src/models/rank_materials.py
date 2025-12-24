import pandas as pd
import joblib

def rank_materials():
    df = pd.read_csv("data/processed/engineered_training_dataset.csv")

    # Load saved objects
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    # Same features as training
    X = df.drop(columns=["recommended"])

    # 🔥 APPLY SAME PREPROCESSING
    X_processed = preprocessor.transform(X)

    # Predict scores
    scores = model.predict_proba(X_processed)[:, 1]

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
