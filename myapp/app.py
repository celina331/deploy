# ---------------------------------
# IMPORTS
# ---------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------
# CONFIG
# ---------------------------------
st.set_page_config(page_title="Titanic ML Project", layout="wide")

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #EDE7F6;
}

section[data-testid="stSidebar"] {
    background-color: #D1C4E9;
}

h1 {
    color: #5E35B1;
}

h2, h3 {
    color: #673AB7;
}

</style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction Dashboard")

st.write("Machine Learning project using Linear Regression")

# ---------------------------------
# LOAD DATA
# ---------------------------------
import os

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "titanic.csv")

    df = pd.read_csv(csv_path)
    return df

df = load_data()

# ---------------------------------
# SIDEBAR FILTERS
# ---------------------------------
st.sidebar.header("Filters")

if "Sex" in df.columns:
    sex_filter = st.sidebar.selectbox("Sex", ["All"] + list(df["Sex"].dropna().unique()))
else:
    sex_filter = "All"

if sex_filter != "All":
    df = df[df["Sex"] == sex_filter]

# ---------------------------------
# DATA PREVIEW
# ---------------------------------
st.header("Dataset Preview")
st.write(df.head())

st.write("Shape:", df.shape)

# ---------------------------------
# PREPROCESSING
# ---------------------------------
st.header("Preprocessing")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df = df.drop(columns=['Cabin', 'Name', 'Ticket'], errors='ignore')

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

st.success("Data cleaned successfully")

# ---------------------------------
# EDA
# ---------------------------------
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival Count")
    st.bar_chart(df['Survived'].value_counts())

with col2:
    st.subheader("Class Distribution")
    st.bar_chart(df['Pclass'].value_counts())

st.subheader("Age Distribution")
fig, ax = plt.subplots()
df['Age'].hist(bins=30, ax=ax)
st.pyplot(fig)

st.subheader("Fare Distribution")
fig2, ax2 = plt.subplots()
df['Fare'].hist(bins=30, ax=ax2)
st.pyplot(fig2)

# ---------------------------------
# CORRELATION HEATMAP
# ---------------------------------
st.header("Correlation")

fig3, ax3 = plt.subplots()
corr = df.corr(numeric_only=True)
cax = ax3.matshow(corr)
fig3.colorbar(cax)
st.pyplot(fig3)

# ---------------------------------
# LINEAR REGRESSION (NUMERICAL ONLY)
# ---------------------------------

# Sélection uniquement colonnes numériques utiles
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

X = df[numeric_features]
y = df['Survived']

# Gestion des valeurs manquantes
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# Split dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# ---------------------------------
# RESULTS VISUALIZATION
# ---------------------------------
st.header("Actual vs Predicted")

fig4, ax4 = plt.subplots()
ax4.scatter(y_test, y_pred)
ax4.set_xlabel("Actual")
ax4.set_ylabel("Predicted")
st.pyplot(fig4)

# ---------------------------------
# CONCLUSION
# ---------------------------------
st.header("Conclusion")

st.write("""
This project analyzes Titanic passenger data and predicts survival using Linear Regression.

Steps:
- Data cleaning
- Feature engineering
- Visualization (EDA)
- Model training
- Evaluation

This dashboard demonstrates a full machine learning pipeline.
""")

st.success("Project Completed Successfully 🚀")
