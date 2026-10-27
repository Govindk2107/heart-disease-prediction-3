import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Dataset me jis class ko "heart disease" maana jaye (usually 1).
# Agar tumhare heart.csv me result ulta aaye to isse 0 kar dena.
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 1 - POSITIVE_CLASS
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart.csv")

# Internet images (Wikimedia Commons). Chalane ke liye internet chahiye.
HERO_IMG = "https://commons.wikimedia.org/wiki/Special:FilePath/Animated_Heart.gif?width=500"
HERO_PAGE = "https://commons.wikimedia.org/wiki/File:Animated_Heart.gif"
ANATOMY_IMG = "https://commons.wikimedia.org/wiki/Special:FilePath/Blausen_0457_Heart_SectionalAnatomy.png?width=800"
ANATOMY_PAGE = "https://commons.wikimedia.org/wiki/File:Blausen_0457_Heart_SectionalAnatomy.png"

FEATURE_NAMES = {
    "age": "Age",
    "sex": "Gender",
    "cp": "Chest Pain Type",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
}
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# ------------------------------------------------------------------ STYLE
st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at 15% 10%, #3b0a2a 0%, #14081f 45%, #070b1a 100%);
}
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3, h4, label, .stMarkdown p, .stCaption, [data-testid="stMetricLabel"] {
    color: #f1f5f9 !important;
}
[data-testid="stMetricValue"] { color: #ffb4c6 !important; }

@keyframes beat {
    0%, 100% { transform: scale(1); }
    20% { transform: scale(1.25); }
    40% { transform: scale(1); }
    60% { transform: scale(1.18); }
}
.beat { display: inline-block; animation: beat 1.3s infinite; }
.hero-title { font-size: 3.1rem; font-weight: 800; line-height: 1.1; margin: 0; }
.grad {
    background: linear-gradient(90deg, #ff758c, #ff7eb3, #ffb199);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle { color: #cbd5e1; font-size: 1.1rem; margin-top: .6rem; }
.author {
    display: inline-block; margin-top: .8rem; padding: .35rem 1rem;
    border-radius: 999px; font-weight: 600; color: #fff;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    box-shadow: 0 4px 18px rgba(255, 65, 108, .45);
}

div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 22px; padding: 1.6rem;
    backdrop-filter: blur(10px);
}
div[data-testid="stFormSubmitButton"] button {
    width: 100%; border: none; border-radius: 14px; padding: .7rem 1rem;
    font-size: 1.1rem; font-weight: 700; color: #fff;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    box-shadow: 0 6px 20px rgba(255, 65, 108, .45);
    transition: transform .15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-2px) scale(1.01); }

.result-card {
    border-radius: 22px; padding: 1.4rem 1.6rem; margin: 1rem 0;
    border: 1px solid rgba(255,255,255,.18); color: #fff;
    backdrop-filter: blur(10px);
}
.result-card h2 { margin: 0 0 .3rem 0; color: #fff !important; }
.risk-high { background: linear-gradient(135deg, rgba(239,68,68,.45), rgba(127,29,29,.55)); }
.risk-low  { background: linear-gradient(135deg, rgba(16,185,129,.40), rgba(6,78,59,.55)); }
@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
.hero-img {
    display: block; margin: 0 auto; width: 100%; max-width: 360px;
    border-radius: 50%; animation: floaty 4s ease-in-out infinite;
    box-shadow: 0 0 60px rgba(255, 65, 108, .55), 0 0 120px rgba(255, 65, 108, .25);
    border: 2px solid rgba(255, 255, 255, .2);
}
.anatomy-img {
    width: 100%; border-radius: 20px; background: #fff; padding: 8px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, .5);
}
.credit { color: #94a3b8; font-size: .8rem; text-align: center; margin-top: .5rem; }
.credit a { color: #ff9ab3; }
.footer { text-align: center; color: #94a3b8; margin-top: 2.5rem; font-size: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ DATA + MODEL
@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


# ------------------------------------------------------------------ 3D VISUALS
def heart_3d():
    """Interactive 3D heart (mouse se ghuma sakte ho)."""
    t = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi, 45)
    T, V = np.meshgrid(t, v)
    hx = 16 * np.sin(T) ** 3
    hy = 13 * np.cos(T) - 5 * np.cos(2 * T) - 2 * np.cos(3 * T) - np.cos(4 * T)
    X = hx * np.sin(V)
    Y = hy * np.sin(V)
    Z = 8 * np.cos(V)

    fig = go.Figure(
        go.Surface(
            x=X, y=Y, z=Z, surfacecolor=Z,
            colorscale=[[0, "#7f1d1d"], [0.5, "#e11d48"], [1, "#ff8fa3"]],
            showscale=False,
            lighting=dict(ambient=0.45, diffuse=0.9, specular=0.9, roughness=0.25),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=0.9, y=0.4, z=1.8)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=340,
    )
    return fig


def risk_gauge(risk):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={"suffix": "%", "font": {"color": "white"}},
            title={"text": "Risk Score", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#ff4b6e"},
                "steps": [
                    {"range": [0, 33], "color": "#14532d"},
                    {"range": [33, 66], "color": "#854d0e"},
                    {"range": [66, 100], "color": "#7f1d1d"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
        height=280, margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


# ------------------------------------------------------------------ LOAD DATA
if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    uploaded = st.file_uploader("heart.csv upload karo", type="csv")
    if uploaded is None:
        st.info("app.py ke same folder me heart.csv rakho ya yahan upload karo.")
        st.stop()
    df = load_data(uploaded)

model, accuracy = train_model(df)
feature_cols = [c for c in df.columns if c != "target"]

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.markdown("## ❤️ Heart Predictor")
st.sidebar.metric("Model accuracy", f"{accuracy * 100:.1f}%")
st.sidebar.metric("Dataset rows", len(df))
st.sidebar.markdown("---")
st.sidebar.caption("By Govind Khatile")

# ------------------------------------------------------------------ HERO
left, right = st.columns([3, 2])
with left:
    st.markdown(
        '<p class="hero-title"><span class="beat">❤️</span> '
        '<span class="grad">Heart Disease<br>Prediction</span></p>'
        '<p class="subtitle">Machine Learning se apne dil ka risk check karo — '
        'sirf kuch details bharo aur result paao.</p>'
        '<span class="author">Developed by Govind Khatile</span>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f'<img class="hero-img" src="{HERO_IMG}" alt="3D animated heart" '
        f'onerror="this.style.display=\'none\'">'
        f'<div class="credit">3D heart: DrJanaOfficial, '
        f'<a href="{HERO_PAGE}" target="_blank">Wikimedia Commons</a> (CC BY-SA 4.0)</div>',
        unsafe_allow_html=True,
    )

tab_predict, tab_3d, tab_heart, tab_about = st.tabs(
    ["🩺 Predict", "📊 3D Data Explorer", "🫀 Heart Anatomy", "ℹ️ About"]
)

# ------------------------------------------------------------------ TAB 1: PREDICT
with tab_predict:
    st.subheader("Patient ki information")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (umar, saal me) — age", 1, 120, 50)

            sex = st.selectbox(
                "Gender (ling) — sex",
                options=[1, 0],
                format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
            )

            cp = st.selectbox(
                "Chest Pain Type (seene ka dard) — cp",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 - Typical Angina",
                    1: "1 - Atypical Angina",
                    2: "2 - Non-anginal Pain",
                    3: "3 - Asymptomatic (dard nahi)",
                }[x],
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg) — trestbps", 50, 250, 120
            )

            chol = st.number_input("Cholesterol (mg/dl) — chol", 100, 600, 240)

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl — fbs",
                options=[0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            )

            restecg = st.selectbox(
                "Resting ECG Result — restecg",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 - Normal",
                    1: "1 - ST-T wave abnormality",
                    2: "2 - Left ventricular hypertrophy",
                }[x],
            )

        with col2:
            thalach = st.number_input(
                "Max Heart Rate Achieved (heart rate, bpm) — thalach", 60, 220, 150
            )

            exang = st.selectbox(
                "Exercise Induced Angina (exercise me seene ka dard) — exang",
                options=[0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            )

            oldpeak = st.number_input(
                "ST Depression (exercise ke baad) — oldpeak",
                0.0, 10.0, 1.0, step=0.1,
            )

            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment — slope",
                options=[0, 1, 2],
                format_func=lambda x: {0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"}[x],
            )

            ca = st.selectbox(
                "Major Vessels coloured by Fluoroscopy (0-4) — ca",
                options=[0, 1, 2, 3, 4],
            )

            thal = st.selectbox(
                "Thalassemia — thal",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 - Null / Unknown",
                    1: "1 - Fixed Defect",
                    2: "2 - Normal",
                    3: "3 - Reversible Defect",
                }[x],
            )

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]],
            columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        )[feature_cols]  # column order dataset jaisa

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        risk = float(proba[list(model.classes_).index(POSITIVE_CLASS)])

        st.session_state["patient"] = input_df.iloc[0].to_dict()
        st.session_state["result"] = (int(pred), risk)

    if "result" in st.session_state:
        pred, risk = st.session_state["result"]
        patient = st.session_state["patient"]

        st.subheader("Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💓 Max Heart Rate", f"{patient['thalach']:.0f} bpm")
        m2.metric("🧪 Cholesterol", f"{patient['chol']:.0f} mg/dl")
        m3.metric("🩸 Resting BP", f"{patient['trestbps']:.0f} mm Hg")
        m4.metric("🎂 Age", f"{patient['age']:.0f}")

        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(risk_gauge(risk))
        with g2:
            if pred == POSITIVE_CLASS:
                st.markdown(
                    f'<div class="result-card risk-high"><h2>⚠️ Risk ZYADA hai</h2>'
                    f"Model ke hisaab se heart disease ka chance <b>{risk * 100:.1f}%</b> hai."
                    f"<br>Kripya kisi cardiologist se zaroor milo.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-card risk-low"><h2>✅ Risk KAM hai</h2>'
                    f"Model ke hisaab se heart disease ka chance <b>{risk * 100:.1f}%</b> hai."
                    f"<br>Phir bhi regular check-up karwate raho.</div>",
                    unsafe_allow_html=True,
                )
            st.caption("Ye ek ML demo hai, medical diagnosis nahi.")

# ------------------------------------------------------------------ TAB 2: 3D EXPLORER
with tab_3d:
    st.subheader("Dataset ka 3D view")
    st.caption("Mouse se ghumao, zoom karo. Agar aapne prediction kiya hai to ⭐ gold diamond aapka patient hai.")

    c1, c2, c3 = st.columns(3)
    xa = c1.selectbox("X axis", NUM_COLS, index=0, format_func=lambda k: FEATURE_NAMES[k])
    ya = c2.selectbox("Y axis", NUM_COLS, index=2, format_func=lambda k: FEATURE_NAMES[k])
    za = c3.selectbox("Z axis", NUM_COLS, index=3, format_func=lambda k: FEATURE_NAMES[k])

    fig3d = go.Figure()
    for cls, name, color in [
        (POSITIVE_CLASS, "Heart Disease", "#ff4b6e"),
        (NEGATIVE_CLASS, "No Disease", "#22d3a5"),
    ]:
        sub = df[df["target"] == cls]
        fig3d.add_trace(
            go.Scatter3d(
                x=sub[xa], y=sub[ya], z=sub[za], mode="markers", name=name,
                marker=dict(size=4, color=color, opacity=0.75),
            )
        )

    if "patient" in st.session_state:
        p = st.session_state["patient"]
        fig3d.add_trace(
            go.Scatter3d(
                x=[p[xa]], y=[p[ya]], z=[p[za]], mode="markers", name="Aapka Patient",
                marker=dict(size=11, color="#facc15", symbol="diamond",
                            line=dict(color="white", width=2)),
            )
        )

    fig3d.update_layout(
        scene=dict(
            xaxis_title=FEATURE_NAMES[xa],
            yaxis_title=FEATURE_NAMES[ya],
            zaxis_title=FEATURE_NAMES[za],
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        height=620, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig3d)

    st.subheader("Kaun si cheez sabse zyada matter karti hai?")
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig_imp = go.Figure(
        go.Bar(
            x=imp.values, y=[FEATURE_NAMES.get(c, c) for c in imp.index],
            orientation="h", marker=dict(color=imp.values, colorscale="Reds"),
        )
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=420, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_imp)

# ------------------------------------------------------------------ TAB 3: HEART ANATOMY
with tab_heart:
    st.subheader("Insaani dil (Heart) ki anatomy")
    a1, a2 = st.columns([3, 2])
    with a1:
        st.markdown(
            f'<img class="anatomy-img" src="{ANATOMY_IMG}" alt="Heart sectional anatomy" '
            f'onerror="this.style.display=\'none\'">'
            f'<div class="credit">Image: Blausen Medical Communications, '
            f'<a href="{ANATOMY_PAGE}" target="_blank">Wikimedia Commons</a> (Creative Commons)</div>',
            unsafe_allow_html=True,
        )
    with a2:
        st.plotly_chart(heart_3d())
        st.caption("🖱️ Is 3D heart ko mouse se ghumao")

# ------------------------------------------------------------------ TAB 4: ABOUT
with tab_about:
    st.subheader("About")
    st.markdown(
        """
- **Model:** Random Forest Classifier (heart.csv par train hua)
- **Input:** 13 medical features (age, cholesterol, heart rate, BP, ECG, etc.)
- **Output:** Heart disease ka risk percentage
- **Images:** Wikimedia Commons se (DrJanaOfficial, CC BY-SA 4.0 aur Blausen Medical Communications, Creative Commons). Credit dena zaroori hai.
- **Note:** Ye project sirf seekhne/demo ke liye hai. Asli medical decision ke liye doctor se milo.
"""
    )

st.markdown('<div class="footer">Made with ❤️ by Govind Khatile</div>', unsafe_allow_html=True)