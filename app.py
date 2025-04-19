import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Smart Route Selector", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    #video-bg {
        position: fixed;
        top: 0;
        left: 0;
        min-width: 100vw;
        min-height: 100vh;
        z-index: -1;
        object-fit: cover;
        opacity: 0.6;
    }

    .stApp {
        background: transparent !important;
    }

    .main-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem auto;
        max-width: 1000px;
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.2);
    }

    h1, h4 {
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 6px #00000088;
    }
    </style>

    <video autoplay loop muted playsinline id="video-bg">
        <source src="https://cdn.pixabay.com/video/2023/07/26/176016-845720984_large.mp4" type="video/mp4">
    </video>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1>🚗 Smart Route Selector</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Your intelligent assistant for personalized travel</h4>", unsafe_allow_html=True)

    purpose = st.selectbox("🎯 What’s your travel purpose?", ["Commute", "Leisure", "Delivery"])
    model_type = st.selectbox("🧠 Choose a Machine Learning Model", ["Random Forest", "SVM", "KNN", "Logistic Regression"])

    def generate_data(n=400, purpose="Commute"):
        distances = np.random.randint(5, 20, size=n)
        avg_speeds = np.random.randint(20, 60, size=n)
        congestion = np.clip(np.random.normal(0.5, 0.2, size=n), 0, 1)
        signals = np.random.randint(2, 10, size=n)

        if purpose == "Commute":
            scores = avg_speeds - distances * 0.6 - congestion * 25 - signals * 1.5
        elif purpose == "Leisure":
            scores = -distances * 0.3 - congestion * 10 - signals * 2
        elif purpose == "Delivery":
            scores = -distances * 0.8 - congestion * 30 - signals * 1.0
        else:
            scores = avg_speeds - distances - congestion * 20 - signals

        return pd.DataFrame({
            'Distance': distances,
            'AvgSpeed': avg_speeds,
            'Congestion': congestion,
            'Signals': signals,
            'BestRoute': np.where(scores < np.percentile(scores, 33), 0,
                                  np.where(scores < np.percentile(scores, 66), 1, 2))
        })

    df = generate_data(purpose=purpose)
    X = df[['Distance', 'AvgSpeed', 'Congestion', 'Signals']]
    y = df['BestRoute']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=1)
    elif model_type == "SVM":
        model = SVC(probability=True)
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=500)

    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    st.subheader("📍 Simulated Congestion Levels")
    fig = px.histogram(df, x='Congestion', nbins=20, color_discrete_sequence=["#FF6F61"])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.2)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛠️ Customize Route Conditions")
    col1, col2 = st.columns(2)
    with col1:
        distance = st.slider("📏 Distance (km)", 5, 20, 10)
        avg_speed = st.slider("🚗 Average Speed (km/h)", 20, 60, 40)
    with col2:
        signals = st.slider("🚦 Number of Signals", 2, 10, 5)
        congestion = st.slider("🌐 Congestion Level (0 - 1)", 0.0, 1.0, 0.5, 0.01)

    user_input = pd.DataFrame([{
        'Distance': distance,
        'AvgSpeed': avg_speed,
        'Congestion': congestion,
        'Signals': signals
    }])

    user_scaled = scaler.transform(user_input)
    probs = model.predict_proba(user_scaled)[0]
    routes = ['Route A', 'Route B', 'Route C']
    prediction = routes[np.argmax(probs)]

    st.success(f"🏁 Best Route: **{prediction}** for **{purpose}** trip")
    st.markdown("### 🔍 Confidence Breakdown")
    st.bar_chart(pd.DataFrame({'Confidence': probs}, index=routes))

    st.markdown(f"<p style='text-align:center;'>Model Accuracy: <b>{accuracy*100:.2f}%</b></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
