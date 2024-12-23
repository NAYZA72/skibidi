import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Linear Regression with Streamlit")

# Input for the number of data points
n_points = st.slider("Number of Data Points", min_value=50, max_value=500, value=100)

# Generate synthetic data based on the selected number of points
np.random.seed(0)
X = 2 * np.random.rand(n_points, 1)
y = 4 + 3 * X + np.random.randn(n_points, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot the data and regression line
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Actual Data')
ax.plot(X_test, y_pred, color='red', label='Regression Line')

ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Linear Regression')

# Display the plot in Streamlit
st.pyplot(fig)

# Optional: Show the model coefficients and intercept
st.write(f"Model Coefficients: {model.coef_[0][0]:.2f}")
st.write(f"Model Intercept: {model.intercept_[0]:.2f}")

