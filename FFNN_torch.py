from matplotlib.pyplot import plot_date
from sklearn.datasets import make_circles, make_classification, make_moons
import matplotlib.animation as ani

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.engine import Events
from skorch import NeuralNetClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split

# TODO: ensemble, dropout/L2 regularization, lr_scheduler

device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device("mps")


epochs = 100

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(inplace=True),
            nn.Linear(6, 6),
            nn.ReLU(inplace=True),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x is not torch.float32:
            x = x.clone().detach().requires_grad_(True).to(dtype=torch.float32)

        logits = self.linear_relu_stack(x)
        logits = torch.flatten(logits)
        return logits



class SamplesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        # self.data = torch.tensor(data, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.float32)

        self.data = data
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    


#################################################################

import torch.optim as to
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


#! Data
# X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)

# X, y = make_moons(n_samples=1000, noise=0.05)

X, y = make_circles(n_samples=1000, noise=0.05, factor=0.3)



def plot_data(X, y):
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y)    
    plt.show()

# plot_data(X, y)



X = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.float32, device=device)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)




train_loader = DataLoader(SamplesDataset(X, y), batch_size = 50)
# test_loader = DataLoader(SamplesDataset(X_test, y_test), batch_size = 50)





#! Instantiate model & trainer
model = FeedForward().to(device)
optimizer = to.Adam(model.parameters(), lr=0.002)
criterion = nn.BCELoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)





#! Grid Search (hyperparameter optimization)
#* params: ['module', 'criterion', 'optimizer', 'lr', 'max_epochs', 'batch_size', 'iterator_train', 'iterator_valid', 'dataset', 'train_split', 'callbacks', 
#* 'predict_nonlinearity', 'warm_start', 'verbose', 'device', 'compile', '_params_to_validate', 'classes']

# param_grid = {
#     'batch_size': [10, 20, 40, 60, 80, 100],
#     'max_epochs': [10, 25, 50, 75, 100],
#     'lr': [0.00005, 0.0005, 0.005, 0.05, 0.5],
#     'optimizer': [to.SGD, to.RMSprop, to.Adagrad, to.Adadelta,
#                   to.Adam, to.Adamax, to.NAdam],
#* not all criterions are suitable!
    # 'criterion': [nn.MSELoss, nn.CrossEntropyLoss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.KLDivLoss],
# }

#* use scorch to connect torch & keras api
# model = NeuralNetClassifier(
#     module=FeedForward,
#     criterion = nn.BCELoss,
#     verbose=False
# )


#* Best: 1.000000 using {'batch_size': 10, 'lr': 0.005, 'max_epochs': 75, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, error_score='raise')

# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# print(X.shape)
# print(y.shape)

#* run grid search
# grid_result = grid.fit(X, y)

#* summarise
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))






#! Evaluation
#* convert
def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


val_metrics = {
    "accuracy": Accuracy(thresholded_output_transform),
    "loss": Loss(criterion)
}


train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
# test_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


#* save all output predictions
# eos = EpochOutputStore()
# eos.attach(train_evaluator, 'total')
# eos.attach(test_evaluator, 'total')


# TODO: attach confusion matrix
# cm = ConfusionMatrix(2)
# cm.attach(train_evaluator, 'cm')


#! Custom logs

#* save info
# tb_logger = TensorboardLogger()

# tb_logger.attach_output_handler(
#     trainer,
#     event_name=Events.ITERATION_COMPLETED(every=500),
#     tag="training",
#     output_transform=lambda loss: {"batch_loss": loss},
# )


#* print loss of each batch
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}], Loss: {engine.state.output:.2f}")

# trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)


#* store losses & accuracy for each epoch
total_train_loss = []
total_train_acc = []


#* print accuracy & average loss every epoch
def log_training_results(trainer):

    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics

    total_train_loss.append(metrics['loss'])
    total_train_acc.append(metrics['accuracy'])
    # print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)



total_test_loss = []
total_test_acc = []

#* same for test/evaluation (using media)
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_validation_results(trainer):
    
#     test_evaluator.run(test_loader)
#     metrics = test_evaluator.state.metrics

#     total_test_loss.append(metrics['loss'])
#     total_test_acc.append(metrics['accuracy'])
    # print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")



#! animate decision boundary evolution over epochs
grid_num = 200



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

xlim, ylim, fig1, ax1 = plot_points(X, y)



def create_grid(xlim, ylim):
    xx, yy = np.meshgrid(np.linspace(*xlim, num=grid_num), np.linspace(*ylim, num=grid_num))
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = torch.tensor(np.hstack((r1,r2)), dtype=torch.float32)

    return xx, yy, grid

xx, yy, grid = create_grid(xlim, ylim)





#* save grid predictions for every epoch
grid_preds = np.empty((epochs, grid_num, grid_num))

def predict_grid(trainer):
    model.eval()
    model.to('cpu')

    preds = model(grid).detach().numpy()
    preds = preds.reshape((grid_num, grid_num))

    grid_preds[trainer.state.epoch-1] = preds

    model.to(device)
    model.train()


trainer.add_event_handler(Events.EPOCH_COMPLETED, predict_grid)


#! Train the model
trainer.run(train_loader, max_epochs=epochs)


#! Save data
file = 'info/'

# np.save(file + 'grid_preds', grid_preds)
# np.save(file + 'losses', total_train_loss)
# np.save(file + 'acc', total_train_acc)

# torch.save(model.state_dict(), 'trained_state.pt')



# grid_preds = np.load(file + 'grid_preds.npy')
# model.load_state_dict(torch.load('trained_state.pt'))
# model.eval()




#! Plotting

#* plot data points with colour-coded predictions
def plot_data_preds():
    output = train_evaluator.state.total
    output = np.array([tup[0].tolist() for tup in output]).reshape(-1)
    output = np.round(output)

    plot_data(X_test, output)




#* plot loss evolution
def plot_loss():
    loss = np.load(file + 'losses.npy')
    acc = np.load(file + 'acc.npy')

    fig2, ax2 = plt.subplots()
    ax2.plot(loss, label='Loss')
    ax2.plot(acc, label='Accuracy')

    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epochs')
    plt.title('Loss/Accuracy over epochs')
    plt.legend()

# plot_loss()



def plot_decision_boundary(grid_preds):

    #* make predictions for each point of the grid

    #* plot decision boundaries
    contours = ax1.contourf(xx, yy, grid_preds, alpha=0.3,
                           levels=np.arange(2 + 1) - 0.5,
                           cmap='viridis',
                           zorder=1)

plot_decision_boundary(grid_preds[-1])




#* get grid predictions for each epoch at a time
def render(i):
    ax = plt.gca()
    contours = ax.contourf(xx, yy, grid_preds[i], alpha=0.3,
                           levels=np.arange(2 + 1) - 0.5,
                           cmap='viridis',
                           zorder=1)
    
    # return fig1, contours


# anim = ani.FuncAnimation(fig1, render, frames=range(epochs), interval=50, repeat=True)


plt.show()