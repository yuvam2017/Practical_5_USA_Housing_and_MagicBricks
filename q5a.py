import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
housing_data = pd.read_csv('USA_Housing.csv')

# Split the data into features and target variable
X = housing_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms']]
y = housing_data['Price']

# Least square method using formulae
X = np.c_[np.ones(X.shape[0]), X]
coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Print the coefficients obtained using the formula
print("Coefficients obtained using the formula:", coeffs)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X[:, 1:], y)

# Predict the target variable using the model
y_pred = model.predict(X[:, 1:])

# Plot the predicted vs actual values
plt.scatter(y, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (USA_Housing.csv)')
plt.show()
