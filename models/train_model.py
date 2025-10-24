"""
Train and evaluate Sensory Behavioral Health AI (with MBTI)
===========================================================
Uses the synthetic dataset to train an XGBoost multi-output regressor on
behavioral features + MBTI (one-hot encoded). Saves the model AND the
final feature column list for correct inference later.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# === Load data ===
print("📥 Loading synthetic dataset...")
df = pd.read_csv("synthetic_behavioral_data.csv")

# === Select features (numeric + mbti_type categorical) ===
numeric_features = [
    "sleep_duration", "sleep_quality", "screen_time_min", "social_interactions",
    "exercise_minutes", "time_outdoors", "academic_pressure", "social_support",
    "avg_heart_rate", "heart_rate_variability", "caffeine_intake_mg",
    "hydration_liters", "calendar_busy_hours", "study_focus_time"
]
categorical = ["mbti_type"]

# One-hot encode MBTI
X = pd.get_dummies(df[numeric_features + categorical], columns=categorical, drop_first=False)
feature_columns = X.columns.tolist()

y = df[["stress_score", "anxiety_score", "mood_score", "burnout_risk"]]

# === Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧪 Train: {X_train.shape}, Test: {X_test.shape}")

# === Train model ===
print("🚀 Training model...")
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=220,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred, multioutput="raw_values")
print(f"✅ R²: Stress={r2[0]:.3f} | Anxiety={r2[1]:.3f} | Mood={r2[2]:.3f} | Burnout={r2[3]:.3f}")

# === Save model + feature columns ===
with open("trained_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical,
        },
        f,
    )
print("💾 Saved trained_model.pkl (includes feature schema)")