import numpy as np
import pandas as pd
from collections import Counter

# Euclidean distance function
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

# KNN class
class KNN:
    def __init__(self, k) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

# Defining main function
def main():

    # Load the dataset (change file path as necessary)
    column_names = ['Age', 'Height', 'Weight', 'Gender', 'Genre']
    data = pd.read_csv('C:/Users/crims/Documents/CSU/CSC525/Module2/data.csv', header=None, names=column_names)

    # Split the dataset into features and labels
    X = data[['Age', 'Height', 'Weight', 'Gender']].values
    y = data['Genre'].values

    # Instantiate the KNN classifier
    knn = KNN(3)

    # Fit the model
    knn.fit(X, y)

    # Ask user for input and convert the inputs to float
    age = float(input("Enter age (in years): "))  # Prompt the user to enter age
    height = float(input("Enter height (in inches): "))  # Prompt the user to enter height
    weight = float(input("Enter weight (in lbs): "))  # Prompt the user to enter weight
    gender = float(input("Enter gender (0 for female, 1 for male): "))  # Prompt the user to enter gender

    new_data = np.array([[age, height, weight, gender]])

    # Predict for new data
    predictions = knn.predict(new_data)

    print("Predicted label:", predictions)

if __name__=="__main__":
    main()