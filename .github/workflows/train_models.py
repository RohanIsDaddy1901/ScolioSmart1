# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
import numpy as np

# 1. Load dataset
csv_file = "scoliosis_trend_and_cobb_demo.csv"
df = pd.read_csv(csv_file)

print("Loaded data shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 2. Drop missing rows if any
df = df.dropna()

# 3. Feature columns
feature_cols = [
    "current_cobb",
    "lumbar_gyro",
    "thoracic_gyro",
    "cervical_gyro",
    "wear_time",
    "pressure",
    "age",
    "risser",
]

X = df[feature_cols]

# 4. Targets
y_trend = df["trend_label"]
y_cobb = df["future_cobb"]

# 5. Split for classifier
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_trend, test_size=0.3, random_state=42
)

# 6. Split for regressor
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_cobb, test_size=0.3, random_state=42
)

# 7. Train classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_c, y_train_c)

# 8. Train regressor
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train_r, y_train_r)

# 9. Evaluate classifier (just for you to see)
trend_pred = clf.predict(X_test_c)
print("\n=== Trend classifier report ===")
print(classification_report(y_test_c, trend_pred))

# 10. Evaluate regressor (just for you to see)
cobb_pred = reg.predict(X_test_r)
mae = mean_absolute_error(y_test_r, cobb_pred)
rmse = np.sqrt(np.mean((y_test_r - cobb_pred) ** 2))
print("\n=== Cobb regressor ===")
print("MAE:", mae)
print("RMSE:", rmse)

# 11. Save models
joblib.dump(clf, "trend_classifier.pkl")
joblib.dump(reg, "cobb_regressor.pkl")

print("\nSaved models:")
print("- trend_classifier.pkl")
print("- cobb_regressor.pkl")
