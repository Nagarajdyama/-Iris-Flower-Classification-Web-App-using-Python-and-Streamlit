import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="🌸 Iris Classifier Pro",
    page_icon="🌺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #fff0f5;
        }
        .stButton > button {
            background-color: #8A2BE2;
            color: white;
            font-weight: 600;
            border-radius: 10px;
        }
        .stMetricValue {
            font-size: 28px !important;
            color: #FF1493 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("IRIS.csv")
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    return df, le

df, le = load_data()
X = df.drop('species', axis=1)
y = df['species']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200),
    "AdaBoost": AdaBoostClassifier(n_estimators=100)
}
for model in models.values():
    model.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("🌿 Input Flower Measurements")
st.sidebar.image("https://i.imgur.com/dO4Cjkt.png", use_column_width=True)
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
model_choice = st.sidebar.radio("Choose Classifier", list(models.keys()))

# Main UI
st.title("🌸 Iris Flower Classifier Pro")
st.markdown("An intelligent dashboard to classify Iris flowers and explore data like never before.")

# Tabs
overview, insights, predict, evaluation = st.tabs(["📊 Data Overview", "📈 Visual Insights", "🤖 Predict", "📉 Evaluation"])

with overview:
    st.subheader("🔍 Iris Dataset Snapshot")
    st.dataframe(df, use_container_width=True)
    st.markdown("### 📌 Feature Summary")
    st.dataframe(df.describe().T, use_container_width=True)
    st.markdown("### 🔗 Correlation Matrix")
    fig_corr = px.imshow(df.corr(), text_auto=True, color_continuous_scale="bluered")
    st.plotly_chart(fig_corr, use_container_width=True)

with insights:
    st.subheader("🧠 Visual Relationships")
    st.plotly_chart(px.scatter_matrix(df, dimensions=df.columns[:-1], color=df['species'].astype(str)), use_container_width=True)
    st.subheader("📊 Distribution by Feature")
    for col in df.columns[:-1]:
        st.plotly_chart(px.violin(df, y=col, x='species', color='species', box=True, points="all"), use_container_width=True)

with predict:
    st.subheader("🌼 Predict the Species")
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    model = models[model_choice]
    prediction = model.predict(input_scaled)
    class_name = le.inverse_transform(prediction)[0]
    st.success(f"🌺 The predicted Iris species is: **{class_name.upper()}**")

    st.markdown("### 📊 Prediction Probabilities")
    prob = model.predict_proba(input_scaled)
    prob_df = pd.DataFrame(prob, columns=le.classes_)
    st.plotly_chart(px.bar(prob_df.T, labels={'index': 'Species', 'value': 'Probability'}, title="Prediction Confidence"), use_container_width=True)

with evaluation:
    st.subheader("📉 Model Comparison")
    eval_data = []
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        eval_data.append((name, score))
    eval_df = pd.DataFrame(eval_data, columns=['Model', 'Accuracy'])
    st.plotly_chart(px.bar(eval_df, x='Model', y='Accuracy', color='Model', title="Model Accuracy Comparison"), use_container_width=True)

    st.markdown("### Confusion Matrix")
    y_pred = models[model_choice].predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale='pinkyl',
        x=le.classes_, y=le.classes_, labels=dict(x="Predicted", y="Actual")), use_container_width=True)

    st.markdown("### Classification Report")
    cr_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)).T
    st.dataframe(cr_df, use_container_width=True)

    st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
