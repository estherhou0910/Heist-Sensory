# 🫧 Sensory: Personality-Driven Behavioral Health AI 🫧

University of Rohcester 2025 Heist Hackathon 
Vault Track: MedTech 
    Proactive, personalized mental-wellness support for college students. Sensory detects, understands, and manages stress through AI-driven insights and adaptive weekly plans built around a student's unique personality and lifestyle.

### 🎯 The Challenge: Generalized Wellness Fails Personalized Needs 😔
College students face an unprecedented stress crisis compounded by generic wellness tools. Standard mental health apps offer one-size-fits-all advice, ignoring a critical factor: individual personality. A high-stress score requires different interventions for different people: an Introvert may need alone time, while an Extrovert may need structured social interaction.

Existing solutions fail to predict who is at risk and why, missing the crucial window for proactive prevention and leading to burnout before effective intervention can occur.

### 💡 The Solution: Sensory—Prediction + Prescription 🎯
Sensory is an AI-powered behavioral health companion that moves mental-wellness from reactive treatment to proactive, personalized prevention. We use advanced machine learning to provide a robust diagnosis and a personality-aligned prescription.

### Sensory achieves its personalization by combining:

🧠 Objective Behavioral Data: Sleep time, sleeping quality, exercise time, outdoor time, study time, and social interaction patterns, screentime, app usage.

💬 Self-Reports: User-submitted emotional and lifestyle check-ins.

🌱 MBTI-Based Personalization: Myers Briggs personality test. Utilizing personality type to tailor coping strategies and recommendations. 

## Key Features:
📈 Predictive AI Diagnosis: Early warning for stress, anxiety, mood, and burnout risk (scale 1–5).

🎯 Personality-Tailored Action Plans: Weekly plans adapted based on individual MBTI type and preferred coping style (the "Prescription").

🗓️ Stress Time Calendar: An integrated calendar view to track stress levels on a monthly, daily, and specific time basis.

👥 Optional Peer Support System: Allows users to unlock connections with others of a compatible personality type or similar interests for accountability and motivation.

📊 Visual Progress Dashboard: A clean, intuitive weekly summary to track emotional progress, including Weekly Progress on specific stress time reduction.

🕒 Focus & Recharge Timer: A simple, integrated tool for balancing productivity phases with essential recovery time.
---------------------------------------------------------------------------------------------------------------------------------------------
## 🧠 How Sensory Works: The Multi-Output Predictive Engine
Sensory's core is the BehavioralHealthPredictor class, implementing a multi-output regression strategy to provide a holistic risk assessment.

Model Architecture:
The system uses four separate, specialized XGBoost Regressors (a highly efficient gradient boosting framework) to predict the four key outcomes independently:
    - stress_score
    - anxiety_score
    - mood_score
    - burnout_risk

### Data Flow and Mechanics:
Input & Collect: Students provide daily logs of lifestyle patterns.

Data Preprocessing:

Imputation: Missing data is handled using median imputation for robustness.

Encoding & Scaling: The categorical mbti_type is label-encoded, and all 30 features are standardized using StandardScaler to ensure fair weighting by the model.

Predict & Analyze: The trained XGBoost models analyze 30 distinct features (Sensing, Lifestyle, and Personality) to generate four precise predictions.

Personalize & Recommend: The application logic uses the predicted scores and the user's encoded MBTI profile to generate a tailored, actionable weekly plan that aligns with their unique psychological needs.
---------------------------------------------------------------------------------------------------------------------------------------------
## Steps to set up locally: 
1️⃣ Setup Environment
```bash
git clone https://github.com/yourusername/Heist-Sensory.git
cd Heist-Sensory
python3 -m venv venv
source venv/bin/activate   # (on macOS/Linux)
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Train the Model (optional)
```bash
python models/train_model.py
```
4️⃣ Run the Streamlit App
```bash
streamlit run UI/app_streamlit.py
```

Then visit 👉 [http://localhost:8501](http://localhost:8501)

## Project Structure 
```
Heist-Sensory/
│
├── models/
│   ├── new_model.py              # BehavioralHealthPredictor class(Main Model)
│   ├── model_v1.py               # Original versions 
│   ├── train_model.py            # AI model training script
│   └── trained_model.pkl         # Serialized trained model
│
├── UI/
│   ├── app_streamlit.py          # Streamlit web app
│   └── images/                   # Figma UI mockups
│
├── data/
│   └── synthetic_behavioral_data.csv     # generated 1000 data
│
├── requirements.txt
└── README.md
```

