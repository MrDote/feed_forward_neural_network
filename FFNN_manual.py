from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import math



# data = load_diabetes()
# X = data['data']
# y = data['target']



class FFNN(BaseEstimator):
    def __init__(self, n_epochs=1, lr=1, act_fn_hidden='sigmoid'):
        """Feedforward Neural Network

        Args:

        """
        
        # np.random.seed(11)

        #! hyperparams
        self.n_epochs = n_epochs
        self.lr = lr
        self.act_fn_hidden = act_fn_hidden

        #! model params
        self.weights = {}
        self.bias = {}

        self.z = {} # weights * inputs + bias for units in each layer
        self.a = {} # activation function applied to z for units in each layer


        #* cz input technically there, just no operations applied to it
        self.n_layers = 1

        #! init sample data
        self.n_samples = None
        self.n_features = None
    

    def add_FC_layer(self, shape):
        self.weights[self.n_layers] = 1/math.sqrt(shape[0]) * (2*np.random.random(tuple(reversed(shape))) - 1)
        self.bias[self.n_layers] = 1/math.sqrt(shape[0]) * (2*np.random.rand(shape[1]) - 1)

        self.n_layers += 1
            

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)


    def _tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2
    

    def _relu(self, x):
        x[x < 0] = 0
        return x
    

    def _relu_derivative(self, x):
        x = x > 0
        return x.astype(int)


    def sum_squared_error(self, preds, true):
        err = 0.5 * np.power(preds - true, 2)
        return err
    

    #* forward pass for each sample
    def _forward(self, input, label):

        input = np.array(input)

        #* check if number of features is same as input neurons in input layer
        if len(input) != self.weights[1].shape[1]:
            raise Exception('Invalid input size! Need: ' + str(len(input)))


        #* activation_fn(inputs * weights + bias)

        #* iterate over layers
        for i in range(1, self.n_layers):
            self.z[i] = np.dot(input, self.weights[i].T) + self.bias[i]

            if i == self.n_layers - 1:
                input = self.z[i]
            else:
                match self.act_fn:
                    case 'relu':
                        # print(self.z[i])
                        input = self._relu(self.z[i])

                    case 'tanh':
                        input = np.tanh(self.z[i])

            self.a[i] = input

        derror = input - label
        return derror



    #* calculate deltas (small changes) for every layer for all samples -> Batch Gradient Descent
    #* Now: apply the change to the weights straight away; Another implementation: first sum all the deltas, then apply change
    def _back_propagate(self, derror):
        deltas = {}

        #* last layer: error
        deltas[self.n_layers-1] = derror * self.z[self.n_layers-1]
        
        #* the rest of the layers going in reverse
        for i in reversed(range(2, self.n_layers)):
            #* current delta = prev delta * weights * d(active_fn)
            match self.act_fn:
                case 'relu':
                    deltas[i-1] = np.dot(self.weights[i].T, deltas[i]) * self.z[i-1]

                case 'tanh':
                    deltas[i-1] = np.dot(self.weights[i].T, deltas[i]) * self._tanh_derivative(self.z[i-1])


        #* update weights & biases
        for i in range(1, self.n_layers-1):
            self.weights[i] = self.weights[i] - self.lr * np.dot(deltas[i].T, self.a[i].T)
            self.bias[i] = self.bias[i] - self.lr * deltas[i]


    # TODO: implement SGD (use random example for update) & Mini-batch (use a group of random examples for update)


    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        
        self.n_samples, self.n_features = X.shape

        self.total_error = []

        for _ in range(self.n_epochs):
            epoch_error = np.empty((self.n_samples))

            for i, (sample_X, sample_y) in enumerate(zip(X, y)):

                derror = self._forward(sample_X, sample_y)

                self._back_propagate(derror)

                #* error calculation
                epoch_error[i] = self.sum_squared_error(self.a[self.n_layers-1], sample_y)

            self.total_error.append(round(np.mean(epoch_error), 7))



# TODO: fix ReLU
nn = FFNN(50, 0.05, 'tanh')
# nn.add_FC_layer((10, 5))
# nn.add_FC_layer((5, 2))
# nn.add_FC_layer((2, 1))

# print(nn.weights)
# print(nn.bias)



from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = datasets.load_diabetes()
data = pd.DataFrame(df.data)
target = df.target

data = data.iloc[:, :]
scaler = StandardScaler()
data = scaler.fit_transform(data)
target = scaler.fit_transform(target.reshape(-1, 1))


# nn.fit(data, target)

# print(nn.total_error)



# nn.sum_squared_error(data, y)






# vector = np.array([2, 3])  # 1x2 vector
# matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # 4x2 matrix

# # Calculate the dot product
# dot_product = np.dot(vector, matrix.T)

# print("Vector:")
# print(vector)
# print("\nMatrix:")
# print(matrix)
# print("\nDot Product:")
# print(dot_product)