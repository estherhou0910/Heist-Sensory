import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sys
import os

# Ensure models module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# ---------- Page & style ----------
st.set_page_config(page_title="Sensory AI", page_icon="🫧", layout="wide")
st.markdown("""
<style>
body {font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial;}
header, footer {visibility: hidden;}
.stMetric {background: #ffffff; border-radius: 16px; padding: 12px;}
.stButton>button {border-radius: 10px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title("🫧 Sensory: Behavioral Health AI (with Personality)")
st.caption("Live AI predictions (1–5 scale) + MBTI-aware weekly action plans tailored to your interests and stressors.")

# ---------- Load model ----------
try:
    with open("models/trained_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    try:
        # newer version with predictor object
        predictor = bundle["predictor"]
        model = predictor.models["multioutput"]
    except KeyError:
        # fallback if only model key exists (older saves)
        model = bundle["model"]
    FEATURE_COLUMNS = bundle["feature_columns"]
    NUMERIC_FEATURES = bundle["numeric_features"]
    CATEGORICAL_FEATURES = bundle["categorical_features"]  # ['mbti_type']
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error(f"⚠️ Could not load model: {e}")
    st.stop()

# ---------- Personality dictionaries ----------
MBTI_TRAITS = {
    "INTJ": {"style": "structured solo deep-work", "focus": ["planning", "analysis", "quiet spaces"]},
    "INTP": {"style": "curious, flexible, exploratory", "focus": ["new ideas", "journaling", "unstructured time"]},
    "ENTJ": {"style": "goal-driven, leadership", "focus": ["targets", "accountability", "time-blocking"]},
    "ENTP": {"style": "novelty-seeking, social brainstorming", "focus": ["variety", "debates", "co-working"]},
    "INFJ": {"style": "reflective, purpose-led", "focus": ["journaling", "nature walks", "mindfulness"]},
    "INFP": {"style": "values-oriented, creative", "focus": ["creative expression", "gentle routines"]},
    "ENFJ": {"style": "relational, coaching", "focus": ["peer support", "group goals", "service"]},
    "ENFP": {"style": "playful, big-picture", "focus": ["novelty", "movement", "social challenges"]},
    "ISTJ": {"style": "consistent, methodical", "focus": ["routines", "checklists", "sleep discipline"]},
    "ISFJ": {"style": "supportive, steady", "focus": ["care rituals", "calm spaces", "accountability buddy"]},
    "ESTJ": {"style": "organized, practical", "focus": ["scheduling", "task batching", "metrics"]},
    "ESFJ": {"style": "community, encouragement", "focus": ["study groups", "celebrations", "shared goals"]},
    "ISTP": {"style": "hands-on, independent", "focus": ["short sprints", "movement", "tactile hobbies"]},
    "ISFP": {"style": "creative, gentle pacing", "focus": ["art time", "nature", "phone boundaries"]},
    "ESTP": {"style": "active, spontaneous", "focus": ["competitions", "buddy workouts", "pomodoros"]},
    "ESFP": {"style": "energetic, social", "focus": ["dance/movement", "outdoor time", "positive routines"]},
}

INTEREST_TAGS = [
    "Running","Yoga","Gym","Dance","Cycling","Board games","Video games","Photography","Art",
    "Cooking","Music","Theater","DIY","Gardening","Volunteering","Meditation","Hiking","Podcasts"
]
STRESSOR_TAGS = [
    "Work deadlines","Academic pressure","Relationship issues","Health worries","Social obligations",
    "Perfectionism","Sleep deprivation","Financial concerns","Isolation","Uncertainty","Overwhelm"
]

# ---------- Inputs ----------
left, right = st.columns(2)
with left:
    sleep_duration = st.slider("🛏️ Sleep Duration (hours)", 3.0, 10.0, 7.0)
    sleep_quality = st.slider("💤 Sleep Quality (1–5)", 1, 5, 4)
    screen_time_min = st.slider("📱 Screen Time (min)", 30, 720, 300)
    social_interactions = st.slider("🤝 Social Interactions (0–10)", 0, 10, 4)
    avg_heart_rate = st.slider("❤️ Avg Heart Rate (bpm)", 55, 120, 75)
    heart_rate_variability = st.slider("💓 HRV (ms)", 20, 100, 60)
    exercise_minutes = st.slider("🏃 Exercise Minutes", 0, 120, 45)

with right:
    time_outdoors = st.slider("☀️ Time Outdoors (min)", 0, 300, 90)
    academic_pressure = st.slider("📚 Academic Pressure (1–5)", 1, 5, 3)
    social_support = st.slider("💬 Social Support (1–5)", 1, 5, 3)
    caffeine_intake_mg = st.slider("☕ Caffeine (mg)", 0, 400, 150)
    hydration_liters = st.slider("💧 Hydration (L)", 0.5, 4.0, 2.0)
    calendar_busy_hours = st.slider("🗓️ Busy Hours", 0, 12, 5)
    study_focus_time = st.slider("🧩 Focus Time (hrs)", 0, 8, 3)

st.markdown("---")
p1, p2, p3 = st.columns([1, 2, 2])

with p1:
    mbti = st.selectbox(
        "🧬 MBTI Personality",
        list(MBTI_TRAITS.keys()),
        index=list(MBTI_TRAITS.keys()).index("ESFJ") if "ESFJ" in MBTI_TRAITS else 0
    )

with p2:
    interests = st.multiselect("🎯 Interests / Hobbies (pick a few)", INTEREST_TAGS, default=["Yoga","Music","Volunteering"])

with p3:
    stressors = st.multiselect("⚠️ Common Stressors (optional)", STRESSOR_TAGS, default=["Academic pressure","Sleep deprivation"])

# ---------- Build model input (align with training schema) ----------
def build_feature_frame():
    base = pd.DataFrame([{
        "sleep_duration": sleep_duration,
        "sleep_quality": sleep_quality,
        "screen_time_min": screen_time_min,
        "social_interactions": social_interactions,
        "exercise_minutes": exercise_minutes,
        "time_outdoors": time_outdoors,
        "academic_pressure": academic_pressure,
        "social_support": social_support,
        "avg_heart_rate": avg_heart_rate,
        "heart_rate_variability": heart_rate_variability,
        "caffeine_intake_mg": caffeine_intake_mg,
        "hydration_liters": hydration_liters,
        "calendar_busy_hours": calendar_busy_hours,
        "study_focus_time": study_focus_time,
        "mbti_type": mbti,
    }])

    # one-hot encode mbti and align to training columns
    X = pd.get_dummies(base, columns=["mbti_type"], drop_first=False)
    # Reindex to the full training feature set, fill missing with 0
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return X

# ---------- Recommendation engine (MBTI + interests + scores) ----------
def summarize_scores(stress, anxiety, mood, burnout):
    stress_txt = "🧘 Very low" if stress < 2 else ("🙂 Moderate" if stress < 3.5 else "😣 High")
    anxiety_txt = "🪷 Very low" if anxiety < 2 else ("😐 Moderate" if anxiety < 3.5 else "😰 High")
    mood_txt = "😔 Low" if mood < 2 else ("🙂 Good" if mood < 3.5 else "😄 Great")
    burnout_txt = "✅ Low" if burnout < 2.5 else ("⚠️ Elevated" if burnout < 3.5 else "🔥 High")
    return stress_txt, anxiety_txt, mood_txt, burnout_txt

def pick_interest(preferred, fallbacks):
    # choose the first matching interest from user's list; else a fallback
    for tag in preferred:
        if tag in interests:
            return tag
    return fallbacks[0] if fallbacks else (interests[0] if interests else "Walk")

def generate_action_plan(mbti_code, scores):
    stress, anxiety, mood, burnout = scores
    trait = MBTI_TRAITS.get(mbti_code, {"style":"balanced","focus":["consistency"]})

    plan = []
    # Core pillars chosen by current needs
    if stress >= 3 or anxiety >= 3:
        # stress/anxiety relief
        preferred = ["Yoga","Meditation","Hiking","Running"]
        chosen = pick_interest(preferred, ["Walk"])
        plan.append({
            "title": f"{chosen} + breathing practice",
            "why": "Lower arousal and reset stress response",
            "when": "Daily",
            "duration": "20–30 min",
            "mbti_fit": trait["style"]
        })

    if mood <= 2.5:
        # mood lift
        preferred = ["Music","Dance","Photography","Art","Volunteering"]
        chosen = pick_interest(preferred, ["Music"])
        plan.append({
            "title": f"{chosen} session for mood",
            "why": "Pleasant engagement increases positive affect",
            "when": "3x / week",
            "duration": "30–45 min",
            "mbti_fit": ", ".join(trait["focus"][:2])
        })

    if screen_time_min > 360 or burnout >= 3:
        plan.append({
            "title": "Digital detox block (app limits on phone)",
            "why": "Reduce doom-scrolling & mental fatigue",
            "when": "Evenings",
            "duration": "90–120 min",
            "mbti_fit": "Boundaries & focus"
        })

    # Always include a sleep anchor if quality/duration is low
    if sleep_duration < 6.5 or sleep_quality <= 3:
        plan.append({
            "title": "Sleep wind-down routine",
            "why": "Consistent cueing improves sleep quality",
            "when": "Nightly",
            "duration": "30 min",
            "mbti_fit": "Calm, consistent ritual"
        })

    # If plan is still short, add a general MBTI-aligned focus item
    if len(plan) < 3:
        if mbti_code in ["ESTJ","ENTJ","ISTJ"]:
            plan.append({"title":"Time-blocking tomorrow",
                         "why":"Structure reduces uncertainty stress",
                         "when":"Daily (evening)",
                         "duration":"10–15 min",
                         "mbti_fit":"Planning & structure"})
        elif mbti_code in ["ENFP","ESFP","ENTP"]:
            plan.append({"title":"Social study sprint",
                         "why":"Energy from people + accountability",
                         "when":"2–3x / week",
                         "duration":"45–60 min",
                         "mbti_fit":"Novelty & connection"})
        else:
            plan.append({"title":"Nature walk + reflection",
                         "why":"Restores attention & mood",
                         "when":"Daily",
                         "duration":"15–20 min",
                         "mbti_fit":"Reflection & calm"})

    return plan[:4]

# ---------- Run prediction ----------
if st.button("✨ Run Prediction"):
    X = build_feature_frame()
    preds = model.predict(X)[0]
    stress, anxiety, mood, burnout = preds

    # Clip to 1–5 for clear scale
    stress = float(np.clip(stress, 1, 5))
    anxiety = float(np.clip(anxiety, 1, 5))
    mood = float(np.clip(mood, 1, 5))
    burnout = float(np.clip(burnout, 1, 5))

    st.markdown("## 🧩 Results Summary (out of 5)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧠 Stress", f"{stress:.2f} / 5")
    c2.metric("💭 Anxiety", f"{anxiety:.2f} / 5")
    c3.metric("🌈 Mood", f"{mood:.2f} / 5")
    c4.metric("🔥 Burnout Risk", f"{burnout:.2f} / 5")

    stress_txt, anxiety_txt, mood_txt, burnout_txt = summarize_scores(stress, anxiety, mood, burnout)
    st.caption(f"• Stress: {stress_txt} • Anxiety: {anxiety_txt} • Mood: {mood_txt} • Burnout: {burnout_txt}")

    st.markdown("---")
    st.subheader("📝 Weekly Action Plan (personalized)")

    plan = generate_action_plan(mbti, (stress, anxiety, mood, burnout))

    for i, item in enumerate(plan, 1):
        st.markdown(
            f"**{i}. {item['title']}**  \n"
            f"• *Why:* {item['why']}  \n"
            f"• *When:* {item['when']}  \n"
            f"• *Duration:* {item['duration']}  \n"
            f"• *Personality fit:* {item['mbti_fit']}"
        )

    st.markdown("---")
    st.subheader("🔒 Privacy & Sharing")
    st.caption("Peer support / accountability is **opt-in**. You can export this plan without sharing any scores.")

# ---------- Static UI Preview Section ----------
st.markdown("### 🎨 Figma-Inspired UI Mockups")

tab1, tab2, tab3, tab4 = st.tabs(["Sensory Vision", "Home", "🧭 Action Plan", "Weekly Summary"])

with tab1:
    st.image("UI/images/startpage.png", caption="Sensory Start Page", use_container_width=True)

with tab2:
    st.image("UI/images/ui_home.png", caption="Home Screen — Overview and Stress Insights", use_container_width=True)

with tab3:
    st.image("UI/images/ui_actionplan.png", caption="Weekly Action Plan — Personalized Tasks & Progress Tracker", use_container_width=True)

with tab4:
    st.image("UI/images/ui_weekly_summary.png", caption="Weekly Progress Summary — Trend Analysis Weekly Review & Achievment Progress", use_container_width=True)