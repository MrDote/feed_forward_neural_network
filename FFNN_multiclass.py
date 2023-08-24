import torch
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
from matplotlib import pyplot as plt





class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 3),
            nn.Softmax(dim=1)
        )

        self.logs = []

    def forward(self, x):
        if x is not torch.float32:
            x = x.clone().detach().requires_grad_(True).to(dtype=torch.float32)

        logits = self.linear_relu_stack(x)
        return logits
    


model = NeuralNetClassifier(
    module=FeedForward,
    criterion = nn.CrossEntropyLoss,
    optimizer = torch.optim.Adam,
    max_epochs = 50,
    batch_size = 2,
    train_split = None,
    verbose=False
)


#! DATA

#* For make_multilabel_classification each sample could have associations with more than one class

X, y = make_classification(100, n_features=2, n_classes=3, n_redundant=0, n_clusters_per_class=1, random_state=1)

# print(y)

def plot_data(X, y):
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y)    
    plt.show()

# plot_data(X, y)



X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)



model.fit(X, y)

# preds = model.predict(X)









def plot_points(X, y):
    fig1, ax1 = plt.subplots(figsize=(10,6))

    #* plot points (since don't change)
    X = X.to('cpu')
    y = y.to('cpu')
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y ,cmap='viridis', s=30, zorder=3)
    ax1.axis('tight')
    ax1.axis('on')

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    ax1.set(xlim=xlim, ylim=ylim)

    return xlim, ylim, fig1, ax1



grid_num = 200
def create_grid(xlim, ylim):
    xx, yy = np.meshgrid(np.linspace(*xlim, num=grid_num), np.linspace(*ylim, num=grid_num))
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = torch.tensor(np.hstack((r1,r2)), dtype=torch.float32)

    return xx, yy, grid




def predict_grid(grid):

    preds = model.predict(grid)
    print(preds)
    preds = preds.reshape((grid_num, grid_num))

    return preds




def plot_decision_boundary(grid_preds):

    #* make predictions for each point of the grid

    #* plot decision boundaries
    contours = ax1.contourf(xx, yy, grid_preds, alpha=0.3,
                           levels=np.arange(3 + 1) - 0.5,
                           cmap='viridis',
                           zorder=1)



xlim, ylim, fig1, ax1 = plot_points(X, y)

xx, yy, grid = create_grid(xlim, ylim)

grid_preds = predict_grid(grid)

plot_decision_boundary(grid_preds)

plt.show()