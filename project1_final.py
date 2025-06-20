# Boston Housing Price Prediction using Decision Tree

# Import necessary libraries
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from numpy import sqrt
import pandas as pd


# Step 1: Load and Inspect Dataset


# Load the Boston Housing dataset from OpenML
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Rename target column for clarity
df = df.rename(columns={'MEDV': 'PRICE'})

# Display basic information about the dataset
print("Dataset Info:")
df.info()

# Step 2: Data Preprocessing

# Convert 'RAD' and 'CHAS' to integer types for consistency
df['RAD'] = df['RAD'].astype(int)
df['CHAS'] = df['CHAS'].astype(int)

# Check for any missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 3: Define Features and Target

# Separate independent features (X) and target variable (Y)
X = df.iloc[:, :-1]
Y = df['PRICE']

# Step 4: Train-Test Split

# Split data into training and test sets (80% train, 20% test)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=21)

# Step 5: Train the Model

# Create and configure Decision Tree Regressor
Tree_Classifier = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=8,                            #Done this through GridSearchCV to find best parameters.
    max_features='sqrt',
    random_state=26
)

# Fit model to training data
Tree_Classifier.fit(X_Train, Y_Train)

# Step 6: Make Predictions

Y_Pred = Tree_Classifier.predict(X_Test)

# Step 7: Evaluate the Model

# Calculate evaluation metrics
mae = mean_absolute_error(Y_Test, Y_Pred)
mse = mean_squared_error(Y_Test, Y_Pred)
rmse = sqrt(mse)
r2 = r2_score(Y_Test, Y_Pred)

# Print metrics
print(f"\nModel Evaluation Metrics:")
print(f"Mean Absolute Error  : {mae:.2f}")
print(f"Mean Squared Error   : {mse:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f}")
print(f"R² Score             : {r2:.4f}")

# Step 8: Visualize Predictions

# Plot actual vs predicted prices
plt.figure(figsize=(8,6))
plt.scatter(Y_Test, Y_Pred, color='purple', alpha=0.6)
plt.plot([Y_Test.min(), Y_Test.max()], [Y_Test.min(), Y_Test.max()], '--', color='orange')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()