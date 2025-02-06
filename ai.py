import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Data Preprocessing (Label Encoding)
def preprocess_data(df, fit_encoder=True):
    # Removed 'Motion/Gyro Data' from the list of categorical columns
    categorical_columns = ['Location (IP/GPS)', 'Browser & Version', 'IP Address & Reputation',
                           'Device Type', 'Network Type', 'VPN/Proxy?', 'User Role',
                           'Multi-Location Logins?', 'Impossible Travel?']
    
    if fit_encoder:
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le  # Store encoder
        joblib.dump(le_dict, 'label_encoders.pkl')  # Save encoders
    else:
        le_dict = joblib.load('label_encoders.pkl')  # Load encoders
        for col in categorical_columns:
            df[col] = le_dict[col].transform(df[col])  # Apply stored encoder
    
    # Convert Date & Time to timestamp feature
    df['Date & Time'] = pd.to_datetime(df['Date & Time']).astype(int) // 10**9  # Convert to seconds
    
    return df

# Cross-validation with Stratified KFold
def cross_validate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(accuracies)

# Train AI Model with Advanced Optimization
def train_model(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    
    X = df.drop(['Access Level (Dynamic)', 'File Sensitivity (Dynamic)'], axis=1)
    y_access = LabelEncoder().fit_transform(df['Access Level (Dynamic)'])
    y_file = LabelEncoder().fit_transform(df['File Sensitivity (Dynamic)'])
    
    # Class weight balancing
    rf_access = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_file = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # Cross-validation for access level
    access_accuracy = cross_validate_model(rf_access, X, y_access)
    print("✅ Cross-validated Access Level Accuracy:", access_accuracy)
    
    # Cross-validation for file sensitivity level
    file_accuracy = cross_validate_model(rf_file, X, y_file)
    print("✅ Cross-validated File Sensitivity Level Accuracy:", file_accuracy)
    
    # Final fit on full training data
    rf_access.fit(X, y_access)
    rf_file.fit(X, y_file)
    
    # Save models
    joblib.dump(rf_access, 'access_model.pkl')
    joblib.dump(rf_file, 'file_model.pkl')
    
    # Predictions
    y_access_pred = rf_access.predict(X)
    y_file_pred = rf_file.predict(X)
    
    # Model Accuracy
    print("✅ Access Level Prediction Accuracy:", accuracy_score(y_access, y_access_pred))
    print("✅ File Sensitivity Level Prediction Accuracy:", accuracy_score(y_file, y_file_pred))
    
    # Model Performance
    mse_access = mean_squared_error(y_access, y_access_pred)
    r2_access = r2_score(y_access, y_access_pred)
    print(f"📌 Mean Squared Error (Access Level): {mse_access}")
    print(f"📌 R² Score (Access Level): {r2_access}")
    
    mse_file = mean_squared_error(y_file, y_file_pred)
    r2_file = r2_score(y_file, y_file_pred)
    print(f"📌 Mean Squared Error (File Sensitivity): {mse_file}")
    print(f"📌 R² Score (File Sensitivity): {r2_file}")
    
    print("\n✅ Optimized models trained and saved successfully!")

# Train and Save Optimized Model
train_model('dataset.csv')
