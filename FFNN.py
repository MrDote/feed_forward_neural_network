import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes



# data = load_diabetes()
# X = data['data']
# y = data['target']



class FFNN:
    def __init__(self, iterations=1, learning_rate=1, act_fn_hidden='sigmoid'):
        """Feedforward Neural Network

        Args:

        """
        
        np.random.seed(11)

        #! hyperparams
        self.n_epochs = iterations
        self.lr = learning_rate
        self.act_fn = act_fn_hidden

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
        self.weights[self.n_layers] = 2*np.random.random(tuple(reversed(shape))) - 1
        self.bias[self.n_layers] = 2*np.random.rand(shape[1]) - 1

        self.n_layers += 1
            

    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))
    
    
    def _sigmoid_derivative(self, data):
        return data * (1 - data)


    def _tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2


    def sum_squared_error(self, preds, true):
        err = 0.5 * np.power(preds - true, 2)
        return err
    

    #* forward pass for each sample
    def _forward(self, input, label):

        input = np.array(input)

        #* check if number of features is same as input neurons in input layer
        if len(input) != self.weights[1].shape[1]:
            raise Exception('Invalid input size!')


        #* activation_fn(inputs * weights + bias)

        #* iterate over layers
        for i in range(1, self.n_layers):
            self.z[i] = np.dot(input, self.weights[i].T) + self.bias[i]

            if i == self.n_layers - 1:
                input = self.z[i]
            else:
                match self.act_fn:
                    case 'sigmoid':
                        # print(self.z[i])
                        input = self._sigmoid(self.z[i])

                    case 'tanh':
                        input = np.tanh(self.z[i])

            self.a[i] = input
            output = input

        ff = output - label
        return ff



    #* calculate deltas (small changes) for every layer
    def _back_propagate(self, derror):
        deltas = {}

        #* last layer: error
        deltas[self.n_layers-1] = derror * self.z[self.n_layers-1]
        
        #* the rest of the layers going in reverse
        for i in reversed(range(2, self.n_layers)):
            #* current delta = prev delta * weights * d(active_fn)
            match self.act_fn:
                case 'sigmoid':
                    deltas[i-1] = np.dot(self.weights[i].T, deltas[i]) * self._sigmoid_derivative(self.z[i-1])

                case 'tanh':
                    deltas[i-1] = np.dot(self.weights[i].T, deltas[i]) * self._tanh_derivative(self.z[i-1])


        #* update weights & biases
        for i in range(1, self.n_layers-1):
            self.weights[i] = self.weights[i] - self.lr * np.dot(deltas[i].T, self.a[i].T)
            self.bias[i] = self.bias[i] - self.lr * deltas[i]


    def train(self, X, y):
        X, y = np.array(X), np.array(y)
        
        self.n_samples, self.n_features = X.shape

        self.total_error = []

        for _ in range(self.n_epochs):
            epoch_error = np.empty((self.n_samples))

            for i, (sample_X, sample_y) in enumerate(zip(X, y)):

                derror = self._forward(sample_X, sample_y)

                #* error calculation
                epoch_error[i] = self.sum_squared_error(self.a[self.n_layers-1], sample_y)

                self._back_propagate(derror)

            self.total_error.append(round(np.mean(epoch_error), 7))



nn = FFNN(100, 0.005, 'sigmoid')
nn.add_FC_layer((4, 4))
nn.add_FC_layer((4, 4))
nn.add_FC_layer((4, 4))
nn.add_FC_layer((4, 1))

# print(nn.weights)
# print(nn.bias)



from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = datasets.load_diabetes()
data = pd.DataFrame(df.data)
target = df.target

data = data.iloc[:, :4]

scaler = StandardScaler()
data = scaler.fit_transform(data)
target = scaler.fit_transform(target.reshape(-1, 1))


nn.train(data, target)

print(nn.total_error)





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