"""
Sensory Behavioral Health Predictor
===================================
Trains regression models to predict stress, anxiety, mood,
and burnout risk from lifestyle and sensing data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


class BehavioralHealthPredictor:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.imputer = None
        self.feature_names = None
        self.outcome_names = None
        self.encoder = None

    def _get_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=200, random_state=42)
        elif self.model_type == "gradient_boost":
            return GradientBoostingRegressor(n_estimators=200, random_state=42)
        else:
            return Ridge(alpha=1.0)

    def preprocess_data(self, df, is_training=True):
        df = df.copy()

        # Encode MBTI personality
        if is_training:
            self.encoder = LabelEncoder()
            df["mbti_type"] = self.encoder.fit_transform(df["mbti_type"].astype(str))
        else:
            df["mbti_type"] = self.encoder.transform(df["mbti_type"].astype(str))

        # Define feature columns
        sensing_features = [
            "steps_per_day", "avg_heart_rate", "heart_rate_variability",
            "sleep_duration", "sleep_quality", "screen_time_min",
            "phone_unlocks", "ambient_light_exposure", "time_outdoors",
            "avg_noise_level"
        ]
        lifestyle_features = [
            "calendar_busy_hours", "study_focus_time", "social_interactions",
            "exercise_minutes", "caffeine_intake_mg", "hydration_liters",
            "commute_minutes", "study_breaks", "screen_time_night"
        ]
        personality_features = [
            "mbti_type", "self_reported_stress", "self_reported_mood",
            "motivation_level", "academic_pressure", "social_support"
        ]

        outcomes = ["stress_score", "anxiety_score", "mood_score", "burnout_risk"]

        all_features = sensing_features + lifestyle_features + personality_features
        X = df[all_features]

        if is_training:
            self.imputer = SimpleImputer(strategy="median")
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=all_features)
            self.feature_names = all_features
            self.outcome_names = outcomes
        else:
            X = pd.DataFrame(self.imputer.transform(X), columns=self.feature_names)

        return X, df[outcomes]

    def fit(self, df):
        X, y = self.preprocess_data(df, is_training=True)
        for outcome in self.outcome_names:
            model = self._get_model()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y[outcome])
            self.models[outcome] = model
            self.scalers[outcome] = scaler

            scores = cross_val_score(model, X_scaled, y[outcome], cv=3, scoring="r2")
            print(f"{outcome}: R²={scores.mean():.3f}")

        print("✅ Model training complete.")

    def evaluate(self, df):
        X, y = self.preprocess_data(df, is_training=False)
        results = {}
        for outcome, model in self.models.items():
            X_scaled = self.scalers[outcome].transform(X)
            pred = model.predict(X_scaled)
            results[outcome] = {
                "RMSE": np.sqrt(mean_squared_error(y[outcome], pred)),
                "MAE": mean_absolute_error(y[outcome], pred),
                "R²": r2_score(y[outcome], pred)
            }
        return results

    def predict(self, row_dict):
        X = pd.DataFrame([row_dict])
        X["mbti_type"] = self.encoder.transform(X["mbti_type"].astype(str))
        X = pd.DataFrame(self.imputer.transform(X[self.feature_names]), columns=self.feature_names)
        preds = {outcome: float(self.models[outcome].predict(self.scalers[outcome].transform(X))[0]) for outcome in self.models}
        return preds
    if __name__ == "__main__":
        print("✅ new_model.py ran successfully")