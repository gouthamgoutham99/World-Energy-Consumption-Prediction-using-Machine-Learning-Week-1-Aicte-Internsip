import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("C:\\Users\\thano\\AndroidStudioProjects\\Downloads\\Aicte-Internship\\cleaned_world_energy.csv")
print("Data Loaded Successfully!\n")
print(df.head())

# Step 3: Inspect and clean column names
df.columns = df.columns.str.strip().str.lower()
print("\nCleaned Columns:", df.columns.tolist())

# Step 4: Drop unnecessary columns (if any)
for col in ['entity', 'code']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Step 5: Handle missing values
print("\nMissing values before cleaning:\n", df.isna().sum())
df.fillna(0, inplace=True)
print("\nMissing values after cleaning:\n", df.isna().sum())

# Step 6: Convert numeric columns
for col in df.columns:
    if col not in ['country']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Step 7: Select features and target
# Predicting primary_energy_consumption using year, population, gdp, and energy_per_capita
features = ['year', 'population', 'gdp', 'energy_per_capita']
target = 'primary_energy_consumption'

X = df[features]
y = df[target]

# Step 8: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining and testing data prepared.")

# Step 9: Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2 Score):", r2)

# Step 12: Visualization - Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted World Energy Consumption")
plt.grid(True)
plt.show()

# Step 13: Display feature importance (model coefficients)
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\nFeature Importance:")
print(coefficients)

# Step 14: Combine predictions with country and year for reference
results = df.loc[X_test.index, ['country', 'year', 'primary_energy_consumption']].copy()
results['predicted_consumption'] = y_pred
print("\nSample Prediction Results (Country-wise):")
print(results.head(10))
