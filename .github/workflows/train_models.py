import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("scoliosis_trend_and_cobb_demo.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

# Drop missing rows if any
df = df.dropna()

# Feature columns
feature_cols = [
    "current_cobb",
    "lumbar_gyro",
    "thoracic_gyro",
    "cervical_gyro",
    "wear_time",
    "pressure",
    "age",
    "risser"
]

X = df[feature_cols]

# Targets
y_trend = df["trend_label"]
y_cobb = df["future_cobb"]

# Train/test split for classifier
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_trend, test_size=0.3, random_state=42
)

# Train/test split for regressor
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_cobb, test_size=0.3, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_c, y_train_c)

# Train regressor
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train_r, y_train_r)

# Evaluate classifier
trend_pred = clf.predict(X_test_c)

print("\n===== TREND CLASSIFIER RESULTS =====")
print(classification_report(y_test_c, trend_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test_c, trend_pred))

# Evaluate regressor
cobb_pred = reg.predict(X_test_r)

mae = mean_absolute_error(y_test_r, cobb_pred)
mse = mean_squared_error(y_test_r, cobb_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, cobb_pred)

print("\n===== COBB REGRESSOR RESULTS =====")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# Save models
joblib.dump(clf, "trend_classifier.pkl")
joblib.dump(reg, "cobb_regressor.pkl")

print("\nModels saved:")
print("- trend_classifier.pkl")
print("- cobb_regressor.pkl")
