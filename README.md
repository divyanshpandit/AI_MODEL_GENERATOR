# AutoML XGBoost Flask Web App (Classification & Regression)

This repository contains a Flask-based Machine Learning web application that allows users to upload their own dataset (CSV), train an XGBoost model (for either classification or regression), and make predictions on new data using the trained model. The app also offers the option to save and download the trained model for future use.

---

## Features

- Upload your CSV dataset for training.
- Choose between **Classification** or **Regression** model.
- Automatically splits the dataset into training and testing sets.
- Applies **StandardScaler** for feature scaling.
- Trains an **XGBoost Classifier** (for classification) or **XGBoost Regressor** (for regression).
- Displays:
  - Classification: **Accuracy Score**, **Confusion Matrix**.
  - Regression: **Mean Squared Error (MSE)**, **R2 Score**.
- Option to save and download the trained model (as `.json` format).
- Predict new data by uploading a CSV file containing only feature columns.
- Download the predicted results in CSV format.

---

## Model Details

### 1. **Classifier Model:**
- Algorithm: **XGBoost Classifier** (`xgb.XGBClassifier`)
- Target: Categorical (Classification tasks)
- Metrics displayed:
  - **Accuracy Score**
  - **Confusion Matrix**
- Model is saved as: `model_classification.json`

---

### 2. **Regressor Model:**
- Algorithm: **XGBoost Regressor** (`xgb.XGBRegressor`)
- Target: Continuous numerical values (Regression tasks)
- Metrics displayed:
  - **Mean Squared Error (MSE)**
  - **R2 Score**
- Model is saved as: `model_regression.json`

---

3. **Steps to Use:**

### 🔹 Training:
- Upload the training dataset CSV.
- Enter the target column name (the output/label column in your CSV).
- Select model type: **Classification** or **Regression**.
- (Optional) Check the "Save Model" box to save the trained model.
- View model performance (Accuracy/MSE/R2 etc).
- Download the trained model if required.

### 🔹 Prediction:
- Upload new data CSV (should have the same features as training data, excluding the target column).
- Predict using the last trained or loaded model.
- Download the predicted CSV.

---

## Notes

- Ensure that the new data CSV for prediction **does NOT contain the target column**.
- Saved models are stored in JSON format (`model_classification.json` or `model_regression.json`).
- Data is automatically scaled using `StandardScaler`.


---

## Author

Made by [Divyanshu] - For educational and practical machine learning deployment purposes.



