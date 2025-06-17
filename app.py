import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

data = load_data()

# -------------- Main App ----------------

def main():
    st.set_page_config(page_title="Employee Attrition App", layout="wide")
    st.title("🔍 Employee Attrition Prediction & Analysis")
    st.markdown("""
        <style>
            .stApp {background-color: #f8f9fa;}
            .block-container {padding: 2rem;}
            h1, h2, h3 {color: #2c3e50;}
        </style>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("📍 Navigate", ["🏠 Introduction", "📊 EDA", "🤖 Modeling", "✅ Conclusion"])

    if page == "🏠 Introduction":
        show_introduction()
    elif page == "📊 EDA":
        show_eda()
    elif page == "🤖 Modeling":
        show_modeling()
    elif page == "✅ Conclusion":
        show_conclusion()

# -------------- Page Functions ----------------

def show_introduction():
    st.header("📌 Project Overview")
    st.markdown("""
    Welcome to the **Employee Attrition Prediction App** powered by IBM's HR Analytics dataset.

    ### 🎯 Purpose:
    - Understand what leads to employee turnover.
    - Use visual insights for data-driven HR strategies.
    - Predict if an employee might leave (simulated prediction for demo).

    ### 📁 Dataset Snapshot:
    - 👥 1470 Employees
    - 🔢 35 Columns
    - 🎯 Target: `Attrition` (Yes/No)
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2820/2820852.png", width=120)
    st.success("Navigate through the app using the sidebar!")

def show_eda():
    st.header("📊 Exploratory Data Analysis")

    st.subheader("🔸 Attrition Proportion")
    counts = data['Attrition'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("🔸 Overtime vs Attrition")
    ot_table = data.groupby('OverTime')['Attrition'].value_counts().unstack()
    st.bar_chart(ot_table)

    st.subheader("🔸 Department-wise Attrition Rate")
    dept_attr = data.groupby(['Department', 'Attrition']).size().unstack().fillna(0)
    st.bar_chart(dept_attr)

    st.subheader("🔸 Custom Scatter Plot")
    numeric_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears']
    x_var = st.selectbox("Select X-axis", numeric_cols, index=0)
    y_var = st.selectbox("Select Y-axis", numeric_cols, index=1)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=data, x=x_var, y=y_var, hue='Attrition', palette={'Yes': 'red', 'No': 'green'}, ax=ax2)
    st.pyplot(fig2)

def show_modeling():
    st.header("🤖 Predict Employee Attrition (Demo)")
    st.markdown("Fill in details to simulate prediction — this demo always returns 'No'.")

    age = st.slider("🧓 Age", 18, 60, 30)
    income = st.slider("💰 Monthly Income", 1000, 20000, 5000, step=100)
    years_company = st.slider("🏢 Years at Company", 0, 40, 5)
    overtime = st.radio("⏱️ OverTime", ["No", "Yes"], horizontal=True)
    department = st.selectbox("🏬 Department", sorted(data['Department'].unique()))
    jobrole = st.selectbox("🧑‍💼 Job Role", sorted(data['JobRole'].unique()))

    if st.button("🔍 Predict Now"):
        # Simulated Output
        result = "No"
        prob = 0.01

        st.success(f"🧾 Predicted Attrition: **{result}**")
        st.info(f"📊 Simulated Probability of Leaving: **{prob:.2f}**")
        st.warning("⚠️ Note: This is a demo. The prediction is hardcoded for project submission.")

def show_conclusion():
    st.header("📌 Conclusion & HR Recommendations")
    st.markdown("""
### 🧠 Insights:
- Overtime, lower salary, and fewer years at the company correlate with higher attrition.
- Certain departments and job roles have more turnover.
- Employees in early career stages are more likely to leave.

### ✅ HR Suggestions:
- Create policies that support work-life balance.
- Address salary gaps for at-risk employees.
- Improve retention with career development programs.

> 📈 This project demonstrates how data science supports HR in reducing employee attrition.
    """)
    st.balloons()

# -------------- Run App ----------------

if __name__ == "__main__":
    main()
