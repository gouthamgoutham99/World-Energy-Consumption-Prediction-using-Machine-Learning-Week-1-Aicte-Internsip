import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Upload Dataset
from google.colab import files
print("cleaned_energy_data.csv")
uploaded = files.upload()

# Step 3: Load Data
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)
print("\n Data Loaded Successfully!")
print(df.head())

# Step 4: Check Columns
print("\n Columns in Dataset:")
print(df.columns.tolist())

# Step 5: Handle Missing Values
df.fillna(0, inplace=True)

# Step 6: Define Features and Target
X = df[['year', 'population', 'gdp', 'energy_per_capita']]
y = df['primary_energy_consumption']

# Step 7: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate Accuracy
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Convert R² score to percentage for clarity
r2_percent = r2 * 100

print("\n Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Model Accuracy (R² Score): {r2_percent:.2f}%")

# Step 11: Predict Future Energy Consumption (Optional)
try:
    future_year = int(input("\nEnter a year to predict energy consumption: "))
    sample = pd.DataFrame({
        'year': [future_year],
        'population': [df['population'].mean()],
        'gdp': [df['gdp'].mean()],
        'energy_per_capita': [df['energy_per_capita'].mean()]
    })
    future_pred = model.predict(sample)
    print(f"\n Predicted Energy Consumption for {future_year}: {future_pred[0]:.2f}")
except Exception as e:
    print("\n Skipping future prediction (no input provided).")
