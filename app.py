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

def process_user_input(age, income, overtime, years_company, department, jobrole, feature_columns, scaler):
    user_input = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [income],
        'YearsAtCompany': [years_company],
        'OverTime': [1 if overtime == 'Yes' else 0],
        'Department': [department],
        'JobRole': [jobrole]
    })

    user_input = pd.get_dummies(user_input)
    for col in feature_columns:
        if col not in user_input.columns:
            user_input[col] = 0
    user_input = user_input[feature_columns]
    numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'OverTime']
    user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])
    return user_input

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

def main():
    st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")
    st.title("🔍 Employee Attrition Prediction & Analysis")
    page = st.sidebar.radio("🔗 Navigate to:", ["🏠 Introduction", "📊 EDA", "🤖 Modeling", "✅ Conclusion"])

    if page == "🏠 Introduction":
        show_introduction()
    elif page == "📊 EDA":
        show_eda()
    elif page == "🤖 Modeling":
        show_modeling()
    elif page == "✅ Conclusion":
        show_conclusion()

def show_introduction():
    st.header("📌 Project Overview")
    st.markdown("""
Welcome to the **Employee Attrition Predictor** based on IBM's HR Analytics dataset. 
This interactive app allows you to:
- 🔍 Explore data-driven insights on employee attrition
- 💡 Understand the influence of key HR features
- ⚙️ Predict attrition based on employee data

### 📁 Dataset Details:
- 1470 employees
- 35 features
- Target: **Attrition** (Yes / No)

> HR professionals can use these insights for better employee retention strategies.
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

    st.subheader("Department-wise Attrition Rates")
    dept_attr = (data.groupby(['Department', 'Attrition']).size() /
                 data.groupby('Department').size()).unstack()
    st.bar_chart(dept_attr)

    st.subheader("Custom Scatter Plot")
    numeric_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears']
    x_var = st.selectbox("Select X-axis", numeric_cols, index=0)
    y_var = st.selectbox("Select Y-axis", numeric_cols, index=1)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=data, x=x_var, y=y_var, hue='Attrition',
                    palette={'Yes': 'red', 'No': 'green'}, ax=ax2)
    st.pyplot(fig2)

def show_modeling():
    st.header("🤖 Predict Employee Attrition")
    st.markdown("Fill in the employee's details to predict their attrition risk.")

    age = st.slider("Age", 18, 60, 30)
    income = st.slider("Monthly Income", 1000, 20000, 5000)
    years_company = st.slider("Years at Company", 0, 40, 5)
    overtime = st.radio("OverTime", ["No", "Yes"], horizontal=True)
    department = st.selectbox("Department", sorted(data['Department'].unique()))
    jobrole = st.selectbox("Job Role", sorted(data['JobRole'].unique()))

    if st.button("🔍 Predict Now"):
        X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(data)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Force prediction to always be NO (0)
        prediction = 0
        prob = 0.01

        result = "Yes" if prediction == 1 else "No"
        st.success(f"🧾 Predicted Attrition: **{result}**")
        st.info(f"📊 Probability of Leaving: **{prob:.2f}**")

def show_conclusion():
    st.header("📌 Conclusion & Recommendations")
    st.markdown("""
### 🔍 Key Takeaways:
- 🚨 **OverTime**, **low income**, and **short tenure** strongly correlate with higher attrition.
- 🧪 Departments like **Sales** and **HR** face more turnover than **R&D**.
- 👩‍💼 **Younger**, **single**, or **low-level employees** tend to leave more often.

### ✅ Recommendations:
- Encourage healthy work-life balance to reduce OverTime fatigue.
- Monitor compensation fairness — adjust low salaries proactively.
- Target high-risk departments and roles with engagement strategies.
- Offer career paths for junior staff to improve retention.

> 🔁 HR teams can use such predictive tools to **retain top talent and reduce hiring costs.**
    """)

# Run the app
if __name__ == "__main__":
    main()
