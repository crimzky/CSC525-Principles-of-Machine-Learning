import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset from the relative path. Dataset downloaded from https://www.kaggle.com/datasets/funxexcel/boston-housing-dataset-with-column-names as the load_boston() is depreciated
# Assuming the CSV file is in the same directory as the script
boston_df = pd.read_csv('./housing.csv')  # Use a relative path to the dataset

# Verify dataset and column names
print(boston_df)

# Separate features and target variable
X = boston_df.iloc[:, :-1]  # All columns except the last one (features)
y = boston_df.iloc[:, -1]   # The last column (target variable MEDV)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')