import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def preprocess_data(df):
# Drop constant/uninformative columns
    df = df.drop(['EmployeeCount','EmployeeNumber','StandardHours','Over18'], axis=1)
    # Encode binary features
    df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})
    df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
    df['OverTime'] = df['OverTime'].map({'Yes':1,'No':0})
    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, drop_first=True)
    # Separate target and features
    y = df['Attrition'].values
    X = df.drop('Attrition', axis=1)
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test