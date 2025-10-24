"""
Generate synthetic behavioral dataset for Sensory AI
====================================================
Creates AI-simulated, structured data combining sensing,
lifestyle, and self-report features for college students.
Now tuned for realistic correlations and stronger signals.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
uids = np.arange(1, n + 1)

# --- Sensing features ---
steps_per_day = np.random.normal(7500, 2500, n).clip(1000, 20000)
avg_heart_rate = np.random.normal(75, 10, n).clip(55, 120)
heart_rate_variability = np.random.normal(60, 15, n).clip(20, 100)
sleep_duration = np.random.normal(7, 1.2, n).clip(3, 10)
sleep_quality = np.random.normal(3.5, 1, n).clip(1, 5)
screen_time_min = np.random.normal(320, 120, n).clip(30, 720)
phone_unlocks = np.random.normal(55, 25, n).clip(5, 180)
ambient_light_exposure = np.random.normal(220, 80, n).clip(0, 600)
time_outdoors = np.random.normal(110, 60, n).clip(0, 300)
avg_noise_level = np.random.normal(60, 10, n).clip(30, 90)

# --- Lifestyle & context ---
calendar_busy_hours = np.random.normal(5, 2, n).clip(0, 12)
study_focus_time = np.random.normal(3, 1.5, n).clip(0, 8)
social_interactions = np.random.normal(3.5, 2, n).clip(0, 10)
exercise_minutes = np.random.normal(45, 25, n).clip(0, 120)
caffeine_intake_mg = np.random.normal(160, 80, n).clip(0, 400)
hydration_liters = np.random.normal(2.3, 0.6, n).clip(0.5, 4)
commute_minutes = np.random.normal(20, 15, n).clip(0, 90)
study_breaks = np.random.normal(9, 4, n).clip(0, 25)
screen_time_night = np.random.normal(100, 60, n).clip(0, 240)

# --- Personality & self-report ---
mbti_types = np.random.choice(
    ["INFP", "INFJ", "ENFP", "ENFJ", "ISTJ", "ISFJ", "ESTP", "ENTJ"], n
)
self_reported_stress = np.random.randint(1, 6, n)
self_reported_mood = np.random.randint(1, 6, n)
motivation_level = np.random.randint(1, 6, n)
academic_pressure = np.random.randint(1, 6, n)
social_support = np.random.randint(1, 6, n)

# --- Outcome generation (stronger structure) ---
# Stress rises with screen_time, HR, busy schedule, poor sleep, low support
stress_base = (
    0.4 * (screen_time_min / 600)
    + 0.35 * (avg_heart_rate / 100)
    - 0.3 * (sleep_duration / 8)
    + 0.25 * (calendar_busy_hours / 10)
    - 0.25 * (social_support / 5)
)
# Spike if sleep <6h or social <2
stress_base += 0.5 * ((sleep_duration < 6) | (social_interactions < 2))
stress_score = (stress_base * 5 + np.random.normal(0, 0.3, n)).clip(1, 5)

# Anxiety rises with stress, low HRV, high academic pressure, high caffeine
anxiety_base = (
    0.45 * (stress_score / 5)
    + 0.25 * (1 - heart_rate_variability / 100)
    + 0.25 * (academic_pressure / 5)
    + 0.15 * (caffeine_intake_mg / 400)
)
anxiety_score = (anxiety_base * 5 + np.random.normal(0, 0.25, n)).clip(1, 5)

# Mood improves with sleep, outdoors, exercise; declines with stress/screen
mood_base = (
    0.4 * (sleep_quality / 5)
    + 0.25 * (time_outdoors / 300)
    + 0.25 * (exercise_minutes / 120)
    - 0.35 * (stress_score / 5)
    - 0.2 * (screen_time_night / 240)
)
mood_score = (mood_base * 5 + np.random.normal(0, 0.2, n)).clip(1, 5)

# Burnout risk increases when stress high + sleep low + focus long
burnout_risk = (
    (stress_score > 3.5).astype(int)
    + (sleep_duration < 6).astype(int)
    + (study_focus_time > 6).astype(int)
) >= 2
burnout_risk = burnout_risk.astype(int)

# --- Combine ---
data = pd.DataFrame({
    "uid": uids,
    "steps_per_day": steps_per_day,
    "avg_heart_rate": avg_heart_rate,
    "heart_rate_variability": heart_rate_variability,
    "sleep_duration": sleep_duration,
    "sleep_quality": sleep_quality,
    "screen_time_min": screen_time_min,
    "phone_unlocks": phone_unlocks,
    "ambient_light_exposure": ambient_light_exposure,
    "time_outdoors": time_outdoors,
    "avg_noise_level": avg_noise_level,
    "calendar_busy_hours": calendar_busy_hours,
    "study_focus_time": study_focus_time,
    "social_interactions": social_interactions,
    "exercise_minutes": exercise_minutes,
    "caffeine_intake_mg": caffeine_intake_mg,
    "hydration_liters": hydration_liters,
    "commute_minutes": commute_minutes,
    "study_breaks": study_breaks,
    "screen_time_night": screen_time_night,
    "mbti_type": mbti_types,
    "self_reported_stress": self_reported_stress,
    "self_reported_mood": self_reported_mood,
    "motivation_level": motivation_level,
    "academic_pressure": academic_pressure,
    "social_support": social_support,
    "stress_score": stress_score,
    "anxiety_score": anxiety_score,
    "mood_score": mood_score,
    "burnout_risk": burnout_risk
})

data.to_csv("synthetic_behavioral_data.csv", index=False)
print("✅ Generated synthetic dataset: synthetic_behavioral_data.csv")
print("Shape:", data.shape)