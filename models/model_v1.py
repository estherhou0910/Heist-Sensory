"""
Behavioral Health Prediction System
=====================================
Predicts mental health outcomes from passive sensing and EMA data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class BehavioralHealthPredictor:
    """
    Main class for predicting mental health outcomes from behavioral data
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the predictor
        
        Args:
            model_type: 'xgboost', 'random_forest', 'gradient_boost', or 'ridge'
        """
        self.model_type = model_type
        self.models = {}  # Dictionary to store models for each outcome
        self.scalers = {}  # Dictionary to store scalers for each outcome
        self.imputers = {}  # Dictionary to store imputers
        self.feature_names = None
        self.outcome_names = None
        self.demographic_encoders = {}
        
    def _get_model(self, outcome_name):
        """Get the appropriate model based on model_type"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'gradient_boost':
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        else:  # ridge
            return Ridge(alpha=1.0, random_state=42)
    
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess raw data
        
        Args:
            df: DataFrame with all features and outcomes
            is_training: Whether this is training data (fit encoders) or test data (transform only)
        
        Returns:
            X: Feature matrix
            y: Target matrix (if training data)
        """
        df = df.copy()
        
        # Define feature categories
        sensing_features = [
            'activity_duration_on_foot',
            'mean_audio_amplitude',
            'num_outgoing_calls',
            'duration_outgoing_calls',
            'light_mean_amplitude',
            'distance_traveled',
            'activity_duration_vehicle',
            'time_leisure_places',
            'time_others_dorms',
            'time_own_dorm',
            'time_religious_places',
            'avg_phone_unlocks_food_location'
        ]
        
        demographic_features = ['gender', 'race']
        
        # Define outcome variables
        outcome_features = [
            'pam_valence',
            'pam_arousal',
            'gad_nervous',
            'gad_unable_stop_worry',
            'phq_feeling_down',
            'phq_little_interest',
            'time_spent_alone_or_others',
            'worry_what_others_think',
            'pleased_with_appearance',
            'feel_as_smart_as_others',
            'overall_feel_good',
            'mean_time_selfesteem_ema_seconds',
            'feeling_stressed'
        ]
        
        # Handle demographics encoding
        for col in demographic_features:
            if col in df.columns:
                if is_training:
                    self.demographic_encoders[col] = LabelEncoder()
                    df[col] = self.demographic_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories gracefully
                    df[col] = df[col].astype(str)
                    known_classes = set(self.demographic_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else 'Unknown')
                    
                    # Add 'Unknown' to encoder if needed
                    if 'Unknown' not in known_classes:
                        self.demographic_encoders[col].classes_ = np.append(
                            self.demographic_encoders[col].classes_, 'Unknown'
                        )
                    df[col] = self.demographic_encoders[col].transform(df[col])
        
        # Combine all input features
        all_features = sensing_features + demographic_features
        available_features = [f for f in all_features if f in df.columns]
        
        X = df[available_features]
        
        # Handle missing values in features
        if is_training:
            self.imputers['features'] = SimpleImputer(strategy='median')
            X = pd.DataFrame(
                self.imputers['features'].fit_transform(X),
                columns=available_features,
                index=X.index
            )
            self.feature_names = available_features
        else:
            X = pd.DataFrame(
                self.imputers['features'].transform(X),
                columns=available_features,
                index=X.index
            )
        
        # Handle outcomes if training data
        if is_training:
            available_outcomes = [f for f in outcome_features if f in df.columns]
            y = df[available_outcomes]
            self.outcome_names = available_outcomes
            return X, y
        
        return X
    
    def fit(self, df, verbose=True):
        """
        Train models for all outcome variables
        
        Args:
            df: DataFrame with features and outcomes
            verbose: Whether to print progress
        """
        X, y = self.preprocess_data(df, is_training=True)
        
        if verbose:
            print(f"Training {self.model_type} models for {len(self.outcome_names)} outcomes...")
            print(f"Features: {len(self.feature_names)}")
            print(f"Samples: {len(X)}")
        
        # Train a separate model for each outcome
        for outcome in self.outcome_names:
            if verbose:
                print(f"\nTraining model for: {outcome}")
            
            # Get non-missing values for this outcome
            mask = ~y[outcome].isna()
            X_train = X[mask]
            y_train = y[outcome][mask]
            
            if len(y_train) < 10:
                if verbose:
                    print(f"  Skipping {outcome} - insufficient data ({len(y_train)} samples)")
                continue
            
            # Scale features
            self.scalers[outcome] = StandardScaler()
            X_scaled = self.scalers[outcome].fit_transform(X_train)
            
            # Train model
            model = self._get_model(outcome)
            model.fit(X_scaled, y_train)
            self.models[outcome] = model
            
            # Evaluate with cross-validation
            if verbose and len(y_train) >= 20:
                cv_scores = cross_val_score(
                    model, X_scaled, y_train, 
                    cv=min(5, len(y_train)//4),
                    scoring='r2'
                )
                print(f"  R² (CV): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        if verbose:
            print(f"\nTraining complete! {len(self.models)} models trained.")
    
    def predict(self, input_data):
        """
        Make predictions for new data
        
        Args:
            input_data: DataFrame or dict with feature values
        
        Returns:
            Dictionary with predictions for each outcome
        """
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        X = self.preprocess_data(input_data, is_training=False)
        
        predictions = {}
        for outcome, model in self.models.items():
            X_scaled = self.scalers[outcome].transform(X)
            pred = model.predict(X_scaled)
            predictions[outcome] = pred[0] if len(pred) == 1 else pred
        
        return predictions
    
    def evaluate(self, test_df, verbose=True):
        """
        Evaluate model performance on test data
        
        Args:
            test_df: DataFrame with features and outcomes
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary with metrics for each outcome
        """
        # For evaluation, we need to process features and get outcomes separately
        X = test_df.copy()
        
        # Extract outcomes before preprocessing
        y = test_df[self.outcome_names].copy()
        
        # Preprocess only the features
        X = self.preprocess_data(X, is_training=False)
        
        results = {}
        
        if verbose:
            print("\nModel Evaluation")
            print("=" * 60)
        
        for outcome, model in self.models.items():
            mask = ~y[outcome].isna()
            if mask.sum() < 5:
                continue
            
            X_test = X[mask]
            y_test = y[outcome][mask]
            
            X_scaled = self.scalers[outcome].transform(X_test)
            y_pred = model.predict(X_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[outcome] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            }
            
            if verbose:
                print(f"\n{outcome}:")
                print(f"  RMSE: {np.sqrt(mse):.3f}")
                print(f"  MAE:  {mae:.3f}")
                print(f"  R²:   {r2:.3f}")
        
        return results
    
    def get_feature_importance(self, outcome, top_n=10):
        """
        Get feature importance for a specific outcome
        
        Args:
            outcome: Name of the outcome variable
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importances
        """
        if outcome not in self.models:
            raise ValueError(f"No model trained for outcome: {outcome}")
        
        model = self.models[outcome]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None
        
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)


# Example usage with REALISTIC synthetic data
if __name__ == "__main__":
    
    """
    Create synthetic data with TRUE SIGNAL + NOISE
    ===============================================
    This simulates realistic relationships between behavioral data and mental health:
    
    PROTECTIVE FACTORS (associated with BETTER mental health):
    - More physical activity (walking, distance traveled)
    - More social interaction (time with others, calls)
    - More time outside own dorm
    - Regular light exposure
    
    RISK FACTORS (associated with WORSE mental health):
    - Social isolation (more time alone in dorm)
    - Less physical activity
    - Excessive phone use
    - Less social contact
    """
    
    np.random.seed(42)
    n_samples = 500
    
    # ========== STEP 1: Generate base behavioral features ==========
    
    # Physical activity (minutes per day on foot)
    # Range: 0-120 minutes, mean ~30 min
    activity_on_foot = np.random.exponential(30, n_samples)
    
    # Distance traveled (meters per day)
    # Range: 0-15000m, mean ~5000m
    distance = np.random.exponential(5000, n_samples)
    
    # Social interaction: time with others (minutes per day)
    # Range: 0-200 minutes, mean ~45 min
    time_with_others = np.random.exponential(45, n_samples)
    
    # Social isolation: time alone in own dorm (minutes per day)
    # Range: 100-500 minutes, mean ~180 min
    time_in_dorm = np.random.normal(180, 60, n_samples)
    time_in_dorm = np.clip(time_in_dorm, 50, 400)
    
    # Phone communication
    num_calls = np.random.poisson(3, n_samples)
    call_duration = np.random.exponential(120, n_samples)
    
    # Phone usage (proxy for rumination/avoidance)
    phone_unlocks = np.random.poisson(5, n_samples)
    
    # Light exposure (lux, natural light exposure)
    light = np.random.normal(300, 100, n_samples)
    light = np.clip(light, 50, 800)
    
    # Audio amplitude (social environment indicator)
    audio = np.random.normal(50, 15, n_samples)
    
    # Other features (less predictive, mostly noise)
    time_vehicle = np.random.exponential(20, n_samples)
    time_leisure = np.random.exponential(60, n_samples)
    time_religious = np.random.exponential(15, n_samples)
    
    # Demographics
    gender = np.random.choice(['Male', 'Female', 'Non-binary'], n_samples)
    race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples)
    
    # ========== STEP 2: Create TRUE mental health outcomes with SIGNAL ==========
    
    # Create a "true underlying mental health score" based on behavioral patterns
    # Lower score = worse mental health
    # This represents the TRUE relationship we want the model to learn
    
    # PROTECTIVE FACTORS (positive coefficients - improve mental health)
    protective_score = (
        0.020 * activity_on_foot +           # More activity = better mood
        0.0004 * distance +                  # More movement = better mood (reduced from 0.0008)
        0.035 * time_with_others +           # Social connection MOST important (increased from 0.020)
        0.010 * num_calls +                  # Communication helps (increased from 0.005)
        0.005 * light +                      # Light exposure helps (increased from 0.003)
        0.015 * audio                        # Social environment helps (increased from 0.01)
    )
    
    # RISK FACTORS (negative coefficients - worsen mental health)
    risk_score = (
        0.025 * time_in_dorm +               # Isolation worsens mood (increased from 0.012)
        0.035 * phone_unlocks                # Excessive phone use = avoidance/rumination (reduced from 0.08)
    )
    
    # Combine to get underlying mental health (higher = better)
    true_mental_health = protective_score - risk_score
    
    # Standardize to mean=0, std=1 for easier interpretation
    true_mental_health = (true_mental_health - true_mental_health.mean()) / true_mental_health.std()
    
    # ========== STEP 3: Generate outcome variables with noise ==========
    
    # GAD (Generalized Anxiety) - "Feeling nervous"
    # Scale: 1-5 (1=not at all, 5=extremely)
    # INVERSE relationship: better mental health → LOWER anxiety
    gad_nervous_true = 3.5 - 0.8 * true_mental_health  # True signal
    gad_nervous_noise = np.random.normal(0, 0.4, n_samples)  # 30% noise
    gad_nervous = np.clip(gad_nervous_true + gad_nervous_noise, 1, 5)
    
    # GAD - "Unable to stop worrying"
    gad_worry_true = 3.3 - 0.75 * true_mental_health
    gad_worry_noise = np.random.normal(0, 0.45, n_samples)
    gad_worry = np.clip(gad_worry_true + gad_worry_noise, 1, 5)
    
    # PHQ (Depression) - "Feeling down, depressed"
    # INVERSE relationship: better mental health → LOWER depression
    phq_down_true = 3.2 - 0.9 * true_mental_health  # Strong signal
    phq_down_noise = np.random.normal(0, 0.35, n_samples)
    phq_down = np.clip(phq_down_true + phq_down_noise, 1, 5)
    
    # PHQ - "Little interest or pleasure"
    phq_interest_true = 3.1 - 0.85 * true_mental_health
    phq_interest_noise = np.random.normal(0, 0.4, n_samples)
    phq_interest = np.clip(phq_interest_true + phq_interest_noise, 1, 5)
    
    # Stress
    stress_true = 3.4 - 0.7 * true_mental_health
    stress_noise = np.random.normal(0, 0.5, n_samples)  # More noise
    stress = np.clip(stress_true + stress_noise, 1, 5)
    
    # Self-esteem items (POSITIVE relationship: better mental health → higher self-esteem)
    selfesteem_true = 3.0 + 0.6 * true_mental_health
    selfesteem_noise = np.random.normal(0, 0.4, n_samples)
    overall_feel_good = np.clip(selfesteem_true + selfesteem_noise, 1, 5)
    
    pleased_appearance_true = 3.2 + 0.5 * true_mental_health
    pleased_appearance = np.clip(pleased_appearance_true + np.random.normal(0, 0.45, n_samples), 1, 5)
    
    feel_smart_true = 3.3 + 0.55 * true_mental_health
    feel_smart = np.clip(feel_smart_true + np.random.normal(0, 0.4, n_samples), 1, 5)
    
    # Outcomes with MORE noise (harder to predict)
    worry_others_think = np.clip(3.0 - 0.4 * true_mental_health + np.random.normal(0, 0.7, n_samples), 1, 5)
    time_alone_or_others = np.clip(2.5 - 0.3 * true_mental_health + np.random.normal(0, 0.8, n_samples), 1, 5)
    
    # PAM scores (Photographic Affect Meter) - mostly noise, weak signal
    pam_valence = np.clip(0.5 + 0.2 * true_mental_health + np.random.normal(0, 0.9, n_samples), -2, 2)
    pam_arousal = np.clip(0 + 0.15 * true_mental_health + np.random.normal(0, 1.0, n_samples), -2, 2)
    
    # Time on self-esteem EMAs (seconds) - noisy
    ema_time = np.clip(np.random.exponential(45, n_samples), 5, 300)
    
    # ========== STEP 4: Combine into DataFrame ==========
    
    data = {
        # Input features
        'activity_duration_on_foot': activity_on_foot,
        'mean_audio_amplitude': audio,
        'num_outgoing_calls': num_calls,
        'duration_outgoing_calls': call_duration,
        'light_mean_amplitude': light,
        'distance_traveled': distance,
        'activity_duration_vehicle': time_vehicle,
        'time_leisure_places': time_leisure,
        'time_others_dorms': time_with_others,
        'time_own_dorm': time_in_dorm,
        'time_religious_places': time_religious,
        'avg_phone_unlocks_food_location': phone_unlocks,
        'gender': gender,
        'race': race,
        
        # Outcome variables (what we're trying to predict)
        'gad_nervous': gad_nervous,
        'gad_unable_stop_worry': gad_worry,
        'phq_feeling_down': phq_down,
        'phq_little_interest': phq_interest,
        'feeling_stressed': stress,
        'overall_feel_good': overall_feel_good,
        'pleased_with_appearance': pleased_appearance,
        'feel_as_smart_as_others': feel_smart,
        'worry_what_others_think': worry_others_think,
        'time_spent_alone_or_others': time_alone_or_others,
        'pam_valence': pam_valence,
        'pam_arousal': pam_arousal,
        'mean_time_selfesteem_ema_seconds': ema_time,
    }
    
    df = pd.DataFrame(data)
    
    # Print the TRUE relationships (so you know what the model should learn)
    print("\n" + "="*60)
    print("TRUE RELATIONSHIPS IN SYNTHETIC DATA")
    print("="*60)
    print("\nPROTECTIVE FACTORS (more = better mental health):")
    print("  ✓ Physical activity (walking)")
    print("  ✓ Distance traveled")
    print("  ✓ Time with others")
    print("  ✓ Number of calls")
    print("  ✓ Light exposure")
    print("  ✓ Audio amplitude (social environment)")
    print("\nRISK FACTORS (more = worse mental health):")
    print("  ✗ Time alone in dorm (isolation)")
    print("  ✗ Phone unlocks (avoidance/rumination)")
    print("\nOUTCOME SIGNALS (expected model performance):")
    print("  Strong signal: PHQ depression, GAD anxiety")
    print("  Moderate signal: Self-esteem, stress")
    print("  Weak signal: PAM scores, social worry")
    print("="*60 + "\n")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize and train model
    predictor = BehavioralHealthPredictor(model_type='xgboost')
    predictor.fit(train_df)
    
    # Evaluate
    results = predictor.evaluate(test_df)
    
    # Make prediction for new data
    new_data = {
        'activity_duration_on_foot': 25,
        'mean_audio_amplitude': 55,
        'num_outgoing_calls': 2,
        'duration_outgoing_calls': 150,
        'light_mean_amplitude': 320,
        'distance_traveled': 4500,
        'activity_duration_vehicle': 15,
        'time_leisure_places': 45,
        'time_others_dorms': 30,
        'time_own_dorm': 200,
        'time_religious_places': 10,
        'avg_phone_unlocks_food_location': 4,
        'gender': 'Female',
        'race': 'Asian'
    }
    
    predictions = predictor.predict(new_data)
    
    print("\n" + "="*60)
    print("PREDICTIONS FOR NEW INDIVIDUAL")
    print("="*60)
    for outcome, value in predictions.items():
        print(f"{outcome}: {value:.2f}")
    
    # Show feature importance for one outcome
    print("\n" + "="*60)
    print("TOP FEATURES FOR STRESS PREDICTION")
    print("="*60)
    if 'feeling_stressed' in predictor.models:
        importance = predictor.get_feature_importance('feeling_stressed')
        print(importance.to_string(index=False))

        