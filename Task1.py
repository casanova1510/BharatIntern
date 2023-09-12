# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing Dataset from the uploaded file
data = pd.read_csv('/content/california_housing_train.csv')  # Adjust the file name if needed

# Split the data into features (X) and target variable (y)
X = data.drop("median_house_value", axis=1)  # Adjust the target variable name if needed
y = data["median_house_value"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a custom linear regression model
class CustomLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.column_stack((np.ones(len(X)), X))
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return X @ self.coefficients + self.intercept

model = CustomLinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Sample new data for demonstration
new_data = pd.DataFrame({
    'MedInc': [3.0],
    'HouseAge': [25.0],
    'AveRooms': [5.0],
    'AveBedrms': [2.0],
    'Population': [1000],
    'AveOccup': [2.5],
    'Latitude': [36.5],
    'Longitude': [-119.5]
})

# Use the trained model to make predictions for the new data
new_predictions = model.predict(new_data)

# Print the predicted house prices
print("\nPredicted House Prices:")
for prediction in new_predictions:
    print(f"${prediction:.2f}")
