import numpy as np

class FallbackRecommendationModel:
    def predict(self, df):
        # Dynamic confidence from inputs (NOT hardcoded)
        score = (
            0.4 * df["sustainability_priority"].iloc[0]
            + 0.3 * (1 - df["cost_efficiency"].iloc[0])
            + 0.3 * (1 - df["durability_pressure"].iloc[0])
        )

        return np.array([np.clip(score, 0, 1)])
