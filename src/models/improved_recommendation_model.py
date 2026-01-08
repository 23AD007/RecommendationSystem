import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.etl.feature_engineering import derive_features

class ImprovedRecommendationModel:
    """Improved recommendation model with better accuracy and performance"""

    def __init__(self, model_path="models/recommendation_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        affinity_path = "models/material_affinity.json"

        if os.path.exists(affinity_path):
            with open(affinity_path, "r") as f:
                self.material_affinity = json.load(f)
        else:
            self.material_affinity = {}
            
        try:
            run_id = mlflow.active_run().info.run_id
        except:
            run_id = None

        if run_id:
            try:
                affinity_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="material_affinity/material_affinity.json"
                )

                with open(affinity_path, "r") as f:
                    self.material_affinity = json.load(f)

                print("✅ Loaded material affinity from MLflow")

            except Exception as e:
                print("⚠️ Could not load material affinity:", e)
                


    def preprocess_data(self, df):
        """Advanced data preprocessing with better handling of missing values"""
        df = df.copy()

        # Use enhanced feature engineering instead of manual processing
        df = derive_features(df, imputation_method="advanced")

        # Handle product_category encoding (additional to feature engineering)
        if 'product_category' in df.columns:
            df['product_category'] = df['product_category'].fillna('unknown')
            df['product_category_encoded'] = df['product_category'].map({
                'synthetic_product': 1,
                'electronics': 2,
                'unknown': 0
            }).fillna(0).astype(int)
        else:
            df['product_category_encoded'] = 0

        return df

        # Apply feature engineering
        df = derive_features(df)

        # Add interaction features for better accuracy
        df['fragility_durability_ratio'] = df['fragility_score'] / (df['durability_requirement'] + 0.001)
        df['cost_sustainability_tradeoff'] = df['material_cost'] * (1 - df['sustainability_priority'])

        return df

    def train_model(self, csv_path="data/processed/training_dataset.csv", use_mlflow=True):
        """Train an improved recommendation model"""
        print("Loading and preprocessing training data...")

        # Load data
        df = pd.read_csv(csv_path)

        if "recommended" not in df.columns:
            raise ValueError("Missing target column: recommended")

        # Preprocess data
        df = self.preprocess_data(df)

        # Define features (include new interaction features)
        base_features = CORE_FEATURES + ['product_category_encoded']
        interaction_features = ['fragility_durability_ratio', 'cost_sustainability_tradeoff']
        self.feature_names = base_features + interaction_features

        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        X = df[self.feature_names]
        y = df["recommended"].astype(int)

        print(f"Training with {len(self.feature_names)} features: {self.feature_names}")
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"After SMOTE - Dataset shape: {X_resampled.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            print(f"{name} CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Train on full training set
            model.fit(X_train_scaled, y_train)

            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)

            print(f"{name} Test Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  AUC: {auc:.4f}")

            # Select best model based on F1-score
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name

        self.model = best_model
        self.is_trained = True

        print(f"\nSelected best model: {best_model_name} with F1-score: {best_score:.4f}")

        # Log to MLflow if requested
        if use_mlflow:
            mlflow.set_experiment("Improved_Packaging_Recommendation")
            with mlflow.start_run(run_name=f"improved_{best_model_name}"):
                mlflow.log_param("model_type", best_model_name)
                mlflow.log_param("features_count", len(self.feature_names))
                mlflow.log_param("smote_used", True)
                mlflow.log_param("scaler_used", True)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(best_model, "model")

        # Save model locally
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': best_model_name
        }
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")

        return self.model

    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            print(f"Loaded model: {model_data.get('model_name', 'Unknown')}")
            return True
        return False

    def predict(self, product_data):
        """Make recommendation prediction for product data"""
        if not self.is_trained and not self.load_model():
            raise ValueError("Model not trained and no saved model found")

        # Convert to DataFrame if dict
        if isinstance(product_data, dict):
            df = pd.DataFrame([product_data])
        else:
            df = product_data.copy()

        # Preprocess
        df = self.preprocess_data(df)

        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select features and scale
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        # Get feature importance for explanation
        feature_importance = dict(zip(self.feature_names,
                                    self.model.feature_importances_)) if hasattr(self.model, 'feature_importances_') else {}

    def recommend_materials(self, product_data, top_n=3):
        """Generate material recommendations with confidence scores and reasons"""
        if not self.is_trained and not self.load_model():
            raise ValueError("Model not trained and no saved model found")

        # Convert to DataFrame if dict
        if isinstance(product_data, dict):
            df = pd.DataFrame([product_data])
        else:
            df = product_data.copy()

        # Preprocess
        df = self.preprocess_data(df)

        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select features and scale
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)

        # Get prediction probabilities
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        confidence_score = float(prediction_proba[1])

        # Define material recommendations based on product characteristics
        materials = self._generate_material_recommendations(df.iloc[0], confidence_score)

        # Sort by score and return top_n
        materials_sorted = sorted(materials, key=lambda x: x['score'], reverse=True)[:top_n]

        return {
            'status': 'success',
            'recommendation_level': 'High' if confidence_score > 0.7 else 'Medium' if confidence_score > 0.5 else 'Low',
            'confidence_score': confidence_score,
            'product_score': confidence_score * 0.9,  # Adjusted product score
            'recommendations': materials_sorted
        }

    def _generate_material_recommendations(self, product_features, base_confidence):
        """Generate material recommendations with confidence scores and reasons"""
        materials = []

        # Extract dataset-approved features
        product_category = product_features.get("product_category", "unknown")
        fragility = float(product_features.get("fragility_score", 0.5))
        sustainability = float(product_features.get("sustainability_priority", 0.5))
        durability = float(product_features.get("durability_requirement", 0.5))
        cost = float(product_features.get("material_cost", 50))
        max_cost = float(product_features.get("max_packaging_cost", 100))
        innovation = float(product_features.get("innovation_level", 0))
        # Product-category affinity for materials
        PRODUCT_CATEGORY_AFFINITY = {
            "electronics": {
                "Recycled Cardboard": 1.0,
                "Bamboo Fiber": 0.9,
                "Hemp Fiber": 0.85,
                "Cork": 1.1
            },
            "food": {
                "Recycled Cardboard": 0.9,
                "Bamboo Fiber": 1.1,
                "Hemp Fiber": 1.0,
                "Cork": 0.8
            },
            "glassware": {
                "Recycled Cardboard": 1.1,
                "Bamboo Fiber": 0.9,
                "Hemp Fiber": 0.85,
                "Cork": 1.2
            },
            "pharmaceutical": {
                "Recycled Cardboard": 1.0,
                "Bamboo Fiber": 0.95,
                "Hemp Fiber": 0.9,
                "Cork": 1.1
            },
            "cosmetics": {
                "Recycled Cardboard": 0.95,
                "Bamboo Fiber": 1.1,
                "Hemp Fiber": 1.0,
                "Cork": 1.0
            },
            "household": {
                "Recycled Cardboard": 1.1,
                "Bamboo Fiber": 0.9,
                "Hemp Fiber": 0.95,
                "Cork": 0.9
            }
        }

        # Material capability profiles (NOT reasons)
        material_db = {
            "Recycled Cardboard": {
                "protection": 0.9,
                "sustainability": 0.95,
                "durability": 0.7,
                "cost_factor": 0.8,
                "innovation_fit": 0.4
            },
            "Bamboo Fiber": {
                "protection": 0.85,
                "sustainability": 0.98,
                "durability": 0.8,
                "cost_factor": 0.75,
                "innovation_fit": 0.6
            },
            "Hemp Fiber": {
                "protection": 0.82,
                "sustainability": 0.97,
                "durability": 0.75,
                "cost_factor": 0.8,
                "innovation_fit": 0.5
            },
            "Cork": {
                "protection": 0.95,
                "sustainability": 0.9,
                "durability": 0.6,
                "cost_factor": 0.85,
                "innovation_fit": 0.7
            }
        }

        for material, profile in material_db.items():

            # -----------------------------
            # Score computation (unchanged logic)
            # -----------------------------
            protection_score = profile["protection"] * fragility
            sustainability_score = profile["sustainability"] * sustainability
            durability_score = profile["durability"] * durability
            cost_score = min(1.0, max_cost / (cost * profile["cost_factor"] + 1))
            innovation_score = profile["innovation_fit"] * innovation

            # Base score (unchanged logic)
            base_material_score = (
                0.30 * protection_score +
                0.25 * sustainability_score +
                0.20 * durability_score +
                0.15 * cost_score +
                0.10 * innovation_score
            )

            # Product-category influence
            category_multiplier = (
                self.material_affinity
                .get(product_category, {})
                .get(material, 1.0)
            )

            material_score = base_material_score * category_multiplier


            confidence = int((base_confidence * 0.6 + material_score * 0.4) * 100)

            # -----------------------------
            # FEATURE-AWARE REASONS
            # -----------------------------
            reasons = []
            if product_category == "electronics":
                reasons.append("Suitable for sensitive electronic components")
            if product_category == "food":
                reasons.append("Meets food safety packaging standards") 
            if product_category == "glassware":
                reasons.append("Provides cushioning for fragile glass items")       

            if fragility > 0.7 and profile["protection"] > 0.8:
                reasons.append("Provides strong protection for fragile products")

            if sustainability > 0.7 and profile["sustainability"] > 0.9:
                reasons.append("Aligns well with high sustainability requirements")

            if durability > 0.7 and profile["durability"] > 0.7:
                reasons.append("Meets elevated durability expectations")

            if cost <= max_cost:
                reasons.append("Fits within the specified packaging budget")

            if innovation > 3 and profile["innovation_fit"] > 0.5:
                reasons.append("Supports preference for innovative materials")
            if category_multiplier > 1.05:
                reasons.append(
                    f"Well-suited for {product_category} packaging requirements"
                )

            if not reasons:
                reasons.append("Balances protection, sustainability, and cost constraints")
            if category_multiplier > 1.05:
                reasons.append(
                    f"Historically performs well for {product_category} packaging"
                )
            elif category_multiplier < 0.95:
                reasons.append(
                    f"Less commonly used for {product_category} compared to alternatives"
                )

            materials.append({
                "material": material,
                "score": round(material_score, 3),
                "confidence": confidence,
                "reason": "; ".join(reasons)
            })

        return materials
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "features": self.feature_names,
            "feature_count": len(self.feature_names),
            "model_type": type(self.model).__name__,
            "scaler_used": self.scaler is not None
        }

# Global model instance for API use
recommendation_model = ImprovedRecommendationModel()

def get_recommendation_model():
    """Get or create the global recommendation model"""
    global recommendation_model
    if not recommendation_model.is_trained:
        if not recommendation_model.load_model():
            print("Training new recommendation model...")
            recommendation_model.train_model()
    return recommendation_model

if __name__ == "__main__":
    # Train and test the improved model
    model = ImprovedRecommendationModel()
    model.train_model()

    # Test prediction
    test_data = {
        'product_category': 'electronics',
        'fragility_score': 0.8,
        'sustainability_priority': 0.7,
        'durability_requirement': 0.6,
        'max_packaging_cost': 100,
        'material_cost': 50,
        'innovation_level': 3
    }

    result = model.predict(test_data)
    print("\nTest Prediction:")
    print(f"Recommended: {result['recommended']}")
    print(f"Confidence: {result['confidence']:.4f}")