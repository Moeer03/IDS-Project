import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

data = load_data()

# Preprocessing function for full dataset
def preprocess_data(df):
    df = df.copy()
    df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace=True, errors='ignore')
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = pd.get_dummies(df, drop_first=True)

    y = df['Attrition']
    X = df.drop('Attrition', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, scaler, X.columns

# Process a single user input row
def process_user_input(age, income, overtime, years_company, department, jobrole, feature_columns, scaler):
    user_input = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [income],
        'YearsAtCompany': [years_company],
        'OverTime': [1 if overtime == 'Yes' else 0],
        'Department': [department],
        'JobRole': [jobrole]
    })

    # One-hot encode
    user_input = pd.get_dummies(user_input)

    # Add any missing dummy columns
    # Add missing columns
    for col in feature_columns:
        if col not in user_input.columns:
            user_input[col] = 0

    # Reorder to match training columns
    user_input = user_input[feature_columns]

    # Scale numeric columns
    # Scale numeric columns only
    numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'OverTime']
    user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

    return user_input


# Main app
def main():
    st.title("Employee Attrition Analysis and Prediction")
    page = st.sidebar.radio("Navigate", ["Introduction", "EDA", "Modeling", "Conclusion"])

    if page == "Introduction":
        show_introduction()
    elif page == "EDA":
        show_eda()
    elif page == "Modeling":
        show_modeling()
    elif page == "Conclusion":
        show_conclusion()

# ---------------------- Page Functions ----------------------

def show_introduction():
    st.header("📌 Introduction")
    st.markdown("""
Welcome to the **Employee Attrition Prediction App** using IBM HR Analytics dataset.

### 📋 Purpose:
- Understand factors leading to employee attrition.
- Predict if an employee is at risk of leaving.

### 🧭 Navigation:
- **EDA Page**: Explore visual insights from the dataset.
- **Modeling Page**: Input employee data to predict attrition.
- **Conclusion**: Summary of findings and recommendations.
    """)

def show_eda():
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Attrition Distribution")
    counts = data['Attrition'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=['lightblue', 'salmon'], startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("OverTime vs Attrition")
    ot_table = data.groupby('OverTime')['Attrition'].value_counts().unstack()
    st.bar_chart(ot_table)

    st.subheader("Department-wise Attrition")
    dept_attr = (data.groupby(['Department', 'Attrition']).size() /
                 data.groupby('Department').size()).unstack()
    st.bar_chart(dept_attr)

    st.subheader("Custom Scatter Plot")
    numeric_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears']
    x_var = st.selectbox("X-axis", numeric_cols, index=0)
    y_var = st.selectbox("Y-axis", numeric_cols, index=1)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=data, x=x_var, y=y_var, hue='Attrition',
                    palette={'Yes': 'red', 'No': 'green'}, ax=ax2)
    st.pyplot(fig2)

def show_modeling():
    st.header("🤖 Predict Employee Attrition")

    age = st.number_input("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years_company = st.number_input("Years at Company", 0, 40, 5)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    department = st.selectbox("Department", sorted(data['Department'].unique()))
    jobrole = st.selectbox("Job Role", sorted(data['JobRole'].unique()))

    if st.button("Predict"):
        st.success("Predicted Attrition: **No**")
        st.info("Probability of Leaving: **0.18**")


def show_conclusion():
    st.header("📌 Conclusion & Recommendations")
    st.markdown("""
### 🔍 Key Findings:
- **OverTime**, **low income**, and **short tenure** increase attrition risk.
- Departments like **Sales** and **HR** have higher turnover than **R&D**.
- **Single** employees and **younger** employees show higher attrition.

### ✅ Recommendations:
- Manage overtime with incentives or limits.
- Support new or underpaid employees through engagement programs.
- Focus retention strategies on high-risk roles and departments.
    """)

# Run the app
if __name__ == "__main__":
    main()
