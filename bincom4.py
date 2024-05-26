import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Convert the string data to a pandas DataFrame
from io import StringIO
df = pd.read_csv('slr.csv')

# Step 2: Extract the features and target variable
X = df[['SAT']].values
y = df['GPA'].values

# Step 3: Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Get the slope and intercept of the line
slope = model.coef_[0]
intercept = model.intercept_

# Predict GPA values
y_pred = model.predict(X)

# Step 4: Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.title('SAT Score vs GPA')
plt.legend()
plt.show()

# Print the slope and intercept
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
