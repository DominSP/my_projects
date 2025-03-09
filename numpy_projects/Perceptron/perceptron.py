# Zaimplementuj perceptron

import numpy as np
import random


def Accuracy (poprawne, udzielone):
    for i in range(len(poprawne)):
        if poprawne[i] == udzielone[i] == 1:
            tp+=1
        elif poprawne[i] == udzielone[i] == 0:
            tn+=1
        elif poprawne[i] == 0 and udzielone[i] == 1:
            fp+=1
        elif poprawne[i] == 1 and udzielone[i] == 0:
            fn+=1
    return (tp + tn)/(tp + tn + fp + fn)

correct = np.array([0,1,0,1,1,1,0,0,1,1,0])
predicted = np.array([1,1,0,0,1,1,1,0,0,0,0])

def true_positives(correct, predicted):
    return ((correct == 1) & (predicted == 1)).sum() #macierze łaczymy bitowymi symb, sum zliczy true jako 1
    #((correct) & (predicted)).sum()
    #(correct * predicted).sum()

def accuracy(correct, predicted):
    return(correct == predicted).sum() / correct.size()

def precision(correct, predicted):
    return 

def recall(correct, predicted):
    return 

def f1(real, predicted, beta):
    prec = precision(real, predicted)
    rec = recall(real, predicted)
    try:
        return (1 + beta**2) * prec * rec / (beta**2 *prec + rec)
    except ZeroDivisionError:
        return 0.0


# def perceptron(x, y): # x wektor wesjciowy, y - prawid. klas.
#     w = np.random.rand(1, 100) #jeden wiersz, 100 kolumn
#     b = 0
#     while epsilon < 0.000001:
#         for d in range(len(x)):
#             a += w[d] * x[d]
#         a += b

class Perceptron:
    def __inti__(self, num_features, epoka = 50, epsilon = 0.00000000001):
        self.weights = np.random.ran(num_features)
        self.bias = np.random.rand(1)[0]

    def predict(self, x):
        return x @ self.weights + self.bias > 0
    
    def train(self, x_train, y_train):
        iter = 0
        while iter < self.epoka:

            for x, y in zip(x_train, y_train):
                if self.predict(x) != y:

                    new_weights = self.weights + y * x
                    if abs( new_weights - self.weights ) > self.epsilon :
                        self.weights = new_weights #*learning rate?
                    self.bias += y #*learning rate?

            iter += 1
        
        return np.array(self.weights, self.bias)
            # az nie wyczerpiemy czasu, lub beda male zmiany w wagach lub nie osiagniemy zadowolajacego bledu na zbiorze treningowym
            # ustalic epoke lub próg tolerancji dla błedu klasyfikacji
# co ma zwrócić train?

    #zrobic przewidywacnie etykiety za pomocą wytrenowanych wag

