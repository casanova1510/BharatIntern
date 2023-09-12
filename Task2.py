import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

wine_data = pd.read_csv('/content/WineQT.csv')

X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

lmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {lmse}")
print(f"R-squared (R2) Score: {r2}")

plt.figure(figsize=(10, 6))

y_pred_all = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_all, alpha=0.5)
plt.xlabel("Actual Wine Quality")
plt.ylabel("Predicted Wine Quality")
plt.title("Linear Regression - Actual vs. Predicted Wine Quality")
plt.show()
