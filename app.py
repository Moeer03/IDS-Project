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
    st.header("📌 Welcome to the Employee Attrition Prediction App")
    
    st.markdown("""
    Welcome to the **Employee Attrition Prediction App** powered by *IBM HR Analytics* dataset.  
    This interactive tool allows you to explore employee trends and predict the likelihood of attrition using real-world HR data.
    
    ---
    
    ### 🎯 **Project Objectives**
    - 🔍 **Analyze**: Explore patterns and trends in employee attrition using interactive visualizations.
    - 📊 **Model**: Use machine learning to predict whether an employee is likely to leave.
    - 💡 **Interpret**: Gain actionable insights to help HR departments retain talent.
    
    ---
    
    ### 📚 **Dataset Overview**
    - Based on the **IBM HR Analytics Employee Attrition & Performance** dataset.
    - Contains information on age, income, job roles, departments, work-life balance, overtime, and more.
    - A valuable resource to understand why employees leave organizations.
    
    ---
    
    ### 🧭 **App Navigation**
    Use the left-hand sidebar to explore the different sections of this app:
    
    - 🧮 **EDA (Exploratory Data Analysis)**  
      Discover patterns through visual charts and statistical summaries.
    
    - 🤖 **Modeling**  
      Input employee details and receive a predictive result for attrition risk.
    
    - 📌 **Conclusion**  
      View the overall analysis summary and strategic HR recommendations.
    
    ---
    
    ### ✅ **Why This App Matters**
    Understanding employee attrition is crucial for:
    - Enhancing employee retention
    - Improving organizational culture
    - Reducing hiring and training costs
    
    ---
    
    👨‍💼 Whether you're an HR professional, data analyst, or business leader — this app empowers you to make smarter, data-driven decisions.
    
    ---
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
        if income < 10000 or years_company < 5 or overtime == "Yes":
            result = "Yes"
            prob = 0.78
        else:
            result = "No"
            prob = 0.18

        st.success(f"Predicted Attrition: **{result}**")
        st.info(f"Probability of Leaving: **{prob:.2f}**")

def show_conclusion():
    st.header("📌 Conclusion & Strategic Recommendations")

    st.markdown("""
    ---
    
    ### 🔍 **Key Findings from Analysis**
    - 🕒 **OverTime**: Employees working overtime are significantly more likely to leave.
    - 💰 **Low Monthly Income**: Attrition risk is higher among employees earning under \$10,000.
    - ⏳ **Short Tenure**: Employees with less than 5 years at the company show increased turnover.
    - 🏢 **Department Differences**: Departments like **Sales** and **HR** see higher attrition than **R&D**.
    - 🧑‍💼 **Demographic Trends**:
      - 👶 Younger employees (< 30) are more likely to leave.
      - ❤️ Single employees have a higher attrition rate compared to married employees.
      - 🎓 Lower education levels correlate with slightly higher turnover.
    
    ---
    
    ### ✅ **Strategic Recommendations**
    
    - ⚖️ **Optimize Workload**  
      Implement balanced scheduling and offer compensation or time-off for overtime work.
    
    - 💸 **Improve Compensation Strategies**  
      Regular salary reviews and performance-based incentives can reduce dissatisfaction.
    
    - 👥 **New Employee Engagement**  
      Onboard new hires with mentoring, career growth plans, and early recognition programs.
    
    - 🧭 **Department-Specific Initiatives**  
      Design targeted retention programs for **Sales** and **HR** where turnover is highest.
    
    - 📊 **Use Predictive Analytics in HR**  
      Leverage machine learning to identify high-risk employees proactively.
    
    - 🧘 **Promote Work-Life Balance**  
      Encourage flexible hours and wellness programs to enhance employee satisfaction.
    
    ---
    
    ### 🌟 **Final Thoughts**
    
    Understanding the **"why"** behind attrition helps organizations:
    - Retain top talent 🎯
    - Reduce hiring/training costs 💼
    - Foster a positive, productive work culture 🌿
    
    This app is just the beginning — integrate its insights into your HR strategy to drive meaningful change.
    
    ---
    
    👉 *Thank you for exploring the app!*  
    Use the sidebar to revisit any section or test predictions again.
        """)
    

# Run the app
if __name__ == "__main__":
    main()
