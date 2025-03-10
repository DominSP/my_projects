import numpy as np

# Ocena klasyfikacji binarnej
def classification_metrics(correct, predicted):
    tp = np.sum((correct == 1) & (predicted == 1)) # TruePositive
    tn = np.sum((correct == 0) & (predicted == 0)) # TrueNegative
    fp = np.sum((correct == 0) & (predicted == 1)) # FalsePositive (błąd typu I)
    fn = np.sum((correct == 1) & (predicted == 0)) # FalseNegative (błąd typu II)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1

# Przykladowe dane
correct = np.array([0,1,0,1,1,1,0,0,1,1,0])
predicted = np.array([1,1,0,0,1,1,1,0,0,0,0])

accuracy, precision, recall, f1 = classification_metrics(correct, predicted)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

class Perceptron:
    def __init__(self, num_features, epochs = 500, learning_rate=0.01, epsilon=1e-10):
        self.weights = np.random.rand(num_features)
        self.bias = np.random.rand(1)[0]
        self.epochs = epochs
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def predict(self, x): # funkcja aktywacji
        return np.where(np.dot(x, self.weights) + self.bias >= 0, 1, 0)
    
    def train(self, x_train, y_train):
            for _ in range(self.epochs):
                prev_weights = self.weights.copy()

                for x, y in zip(x_train, y_train):
                    prediction = self.predict(x)
                    update = y - prediction  # -1, 0, lub 1; jeśli 0 to wagi się nie aktualizują, ponieważ była poprawna predykcja
                    
                    self.weights += self.learning_rate * update * x
                    self.bias += self.learning_rate * update

                # Sprawdzenie, czy wagi się stabilizują
                if np.linalg.norm(self.weights - prev_weights) < self.epsilon:
                    break
 
    def test(self, X):
        return (X @ self.weights + self.bias) > 0
