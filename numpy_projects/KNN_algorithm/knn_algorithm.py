import csv
import numpy as np
from enum import Enum

class DistanceMetric(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    CHEBYSHEV = 'chebyshev'
    COSINE = 'cosine'

class KNNClassifier:
    def __init__(self, k=3, distance_metric=DistanceMetric.EUCLIDEAN):
        if not isinstance(distance_metric, DistanceMetric):
            raise ValueError(f"Invalid distance metric. Choose from: {list(DistanceMetric)}")
        
        self.k = k
        self.distance_metric = distance_metric
        self.x_train = np.empty((0,))
        self.y_train = np.empty((0,), dtype=int)

    def train(self, x, y):
        x, y = np.array(x), np.array(y)
        if self.x_train.size == 0:
            self.x_train, self.y_train = x, y
        else:
            self.x_train = np.vstack((self.x_train, x))
            self.y_train = np.hstack((self.y_train, y))

    def _compute_distances(self, x):
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            return np.linalg.norm(self.x_train - x, axis=1)
        
        elif self.distance_metric == DistanceMetric.MANHATTAN:
            return np.sum(np.abs(self.x_train - x), axis=1)
        
        elif self.distance_metric == DistanceMetric.CHEBYSHEV:
            return np.max(np.abs(self.x_train - x), axis=1)
        
        elif self.distance_metric == DistanceMetric.COSINE:
            x_norm = np.linalg.norm(x)
            train_norms = np.linalg.norm(self.x_train, axis=1)
            ratio = np.dot(self.x_train, x) / (train_norms * x_norm)
            return 1 - ratio
        
        else:
            raise ValueError("Unsupported distance metric")

    def predict(self, X):
        X = np.atleast_2d(X)  # Zapewniamy, że X zawsze jest 2D
        if X.shape[1] != self.x_train.shape[1]:
            raise ValueError(f"Input dimension {X.shape[1]} does not match training data dimension {self.x_train.shape[1]}")

        predictions = []
        for x in X:
            distances = self._compute_distances(x)
            k_nearest_index = np.argpartition(distances, self.k)[:self.k]
            labels = self.y_train[k_nearest_index]
            predictions.append(np.argmax(np.bincount(labels)))

        return np.array(predictions) if len(predictions) > 1 else predictions[0]
    
def open_csv(file_path):
    x, y = [], []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            x.append(list(map(float, row[:-1])))
            y.append(int(row[-1]))
    return np.array(x), np.array(y)

x, y = open_csv('dataset1.csv')       

# Testowanie klasyfikatora
knn = KNNClassifier()
knn.train(x, y)
new_data = np.array([1.0,2.0], [3.4, 5.0], [1.0,3.2])
prediction = knn.predict(new_data)
print(f'Prediction for new data {new_data} is {prediction}') # 1 1 1