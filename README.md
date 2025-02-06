# Context-Aware Security AI

## Overview

This project implements a **Context-Aware Security AI** that dynamically adjusts access levels and file sensitivity based on multiple security parameters, including IP reputation, login consistency, motion data, and network type. It utilizes machine learning models trained on security-related datasets to make real-time access control decisions.

## Features

- **Dynamic Access Control:** Adjusts user permissions based on login context.
- **Security AI Model:** Predicts access levels and file sensitivity based on security attributes.
- **Anomaly Detection:** Identifies suspicious logins using multiple parameters.
- **Role-Based Authentication:** Supports role-based security enforcement.
- **Machine Learning Optimization:** Uses `RandomForestClassifier` with stratified k-fold cross-validation.

## Dataset

The dataset contains multiple security-related parameters:

| Column                 | Description                                                   |
|------------------------|---------------------------------------------------------------|
| Date & Time            | Timestamp of login attempt                                    |
| Location (IP/GPS)      | Login location (e.g., "Chennai, India (TCS)")                 |
| Browser & Version      | Browser used (e.g., "Chrome 120.0")                           |
| IP Address & Reputation| IP address and security rating (Safe/Medium Risk/Suspicious)  |
| Device Type            | Device information (e.g., "Windows 11, HP Laptop")            |
| Login Success Rate     | Probability of successful login (60%-100%)                    |
| Motion/Gyro Data       | Device motion status (Stable/Unstable/Erratic)                |
| IP Consistency         | Whether current IP matches previous logins (True/False)       |
| Network Type           | Wi-Fi, Ethernet, VPN, etc.                                    |
| VPN/Proxy?             | Detects if VPN/Proxy is used (True/False)                     |
| User Role              | Role of user (Admin, Employee, Guest, etc.)                   |
| Access Level           | Predicted access level (0-Full, 3-Read-Only, Denied)          |
| File Sensitivity       | File security classification (High, Medium, Low, No Access)   |
| Failed Logins          | Number of failed attempts                                     |
| Multi-Location Logins? | Whether the user logs in from multiple locations simultaneously (True/False) |
| Impossible Travel?     | Detects impossible travel scenarios (True/False)              |

## AI/ML Model Implementation

The `ai.py` script performs the following tasks:

1. **Loads Dataset:** Reads login attempt records.
2. **Preprocesses Data:** Converts categorical values to numerical encoding.
3. **Trains Models:** Uses `RandomForestClassifier` with class weighting to handle imbalanced data.
4. **Cross-Validation:** Performs 5-fold stratified validation to improve generalization.
5. **Model Optimization:** Evaluates accuracy, mean squared error (MSE), and R² scores.
6. **Saves Models:** Stores trained models for deployment.

## Installation & Setup

**Prerequisites**

- **Python 3.x:** Ensure Python is installed on your system. You can download it from the [official website](https://www.python.org/).
- **Required Libraries:** Install the necessary Python libraries using pip:

  ```bash
  pip install pandas numpy joblib matplotlib seaborn scikit-learn
