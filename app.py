import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ✅ Streamlit Page Settings
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="wide")
st.title("💼 Employee Salary Prediction Using Machine Learning")
st.markdown("Predict employee salary based on professional details using a Random Forest Regression Model.")

# ✅ Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Salary.csv")
    return df

df = load_data()

st.sidebar.header("📌 Enter Employee Details")

# ✅ Preprocess Data with Label Encoding
def preprocess_data(df):
    le_dict = {}
    df_encoded = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le
    return df_encoded, le_dict

df_encoded, le_dict = preprocess_data(df)

# ✅ Features and Target
X = df_encoded.drop(['Salary'], axis=1)
y = df_encoded['Salary']

# ✅ Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ✅ Sidebar Input Section
input_data = {}
for col in X.columns:
    if df[col].dtype == 'object':
        input_data[col] = st.sidebar.selectbox(f"{col}:", df[col].unique())
    else:
        min_val, max_val = int(df[col].min()), int(df[col].max())
        input_data[col] = st.sidebar.slider(f"{col}:", min_val, max_val, min_val)

input_df = pd.DataFrame([input_data])

# ✅ Encode Input Data
for col in input_df.columns:
    if col in le_dict:
        input_df[col] = le_dict[col].transform(input_df[col])

# ✅ Predict Salary
if st.sidebar.button("Predict Salary 💰"):
    predicted_salary = model.predict(input_df)[0]
    st.success(f"🎉 Predicted Monthly Salary: ₹ {int(predicted_salary):,}")

# ✅ Dataset Overview
st.markdown("### 📊 Dataset Sample")
st.dataframe(df.head())

# ✅ Model Performance
st.markdown("### ✅ Model Performance Metrics")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")

# ✅ Visualization Plot
st.markdown("### 📈 Actual vs Predicted Salary Plot")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Prediction Line')
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title("Actual vs Predicted Salary")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("✅ Made by Aman Antuley ")
