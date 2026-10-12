import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_california_housing
import pandas as pd


housing = fetch_california_housing(as_frame=True)


df = housing.frame


print(df.head())

print("\nDataset shape:")
print(df.shape)


print("\nColumn names:")
print(df.columns)


print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

plt.figure(figsize=(8, 5))

sns.histplot(df["MedHouseVal"], bins=30, kde=True)

plt.title("Distribution of California House Values")
plt.xlabel("Median House Value (units of $100,000)")
plt.ylabel("Number of Districts")

plt.tight_layout()
plt.show()

# Plot the relationship between income and house value
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="MedInc",
    y="MedHouseVal",
    alpha=0.4
)

plt.title("Median Income vs Median House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value (units of $100,000)")

plt.tight_layout()
plt.show()

# Calculate correlations between numerical columns
correlation = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of California Housing Data")
plt.tight_layout()
plt.show()

# Separate input features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nInput features (X):")
print(X.head())

print("\nTarget values (y):")
print(y.head())

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# Predict house values using the testing data
y_pred = model.predict(X_test)

print("\nFirst 10 actual house values:")
print(y_test.head(10).values)

print("\nFirst 10 predicted house values:")
print(y_pred[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R² Score:", round(r2, 4))

# Convert MAE and RMSE into dollars
print("MAE in dollars: $", round(mae * 100000, 2))
print("RMSE in dollars: $", round(rmse * 100000, 2))

from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict house values
rf_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest Evaluation Results ---")
print("MAE:", round(rf_mae, 4))
print("MSE:", round(rf_mse, 4))
print("RMSE:", round(rf_rmse, 4))
print("R² Score:", round(rf_r2, 4))
print("MAE in dollars: $", round(rf_mae * 100000, 2))
print("RMSE in dollars: $", round(rf_rmse * 100000, 2))

# Compare both models in one table
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, rf_mae],
    "MSE": [mse, rf_mse],
    "RMSE": [rmse, rf_rmse],
    "R2 Score": [r2, rf_r2]
})

print("\n--- Model Comparison Table ---")
print(comparison.round(4).to_string(index=False))

import pickle

# Save the trained Random Forest model
with open("best_house_price_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("\nRandom Forest model saved successfully!")

# Load the saved model
with open("best_house_price_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Take one example from the test dataset
sample_house = X_test.iloc[[0]]

# Predict the house value
predicted_value = loaded_model.predict(sample_house)[0]

# Convert the prediction into dollars
predicted_dollars = predicted_value * 100000

print("\n--- Saved Model Test ---")
print("Predicted house value:", round(predicted_value, 4))
print("Predicted value in dollars: $", round(predicted_dollars, 2))
