from matplotlib.pyplot import plot_date
from sklearn.datasets import make_classification, make_moons

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.engine import Events

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# TODO: cross validation, ensemble, dropout/L2, grid search

device = 'cpu'
# if torch.backends.mps.is_available():
#     device = torch.device("mps")

epochs = 1


# TODO: lr_scheduler

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        logits = torch.flatten(logits)
        return logits



class SamplesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data).type(torch.float32)
        self.labels = torch.tensor(labels).type(torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    


#################################################################

from torch.optim import Adam
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


#! Data
# X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)

X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)

# X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_data(X, y):
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y)    
    plt.show()




train_loader = DataLoader(SamplesDataset(X, y), 50)


#! Instantiate model & trainer
model = FeedForward().to(device)
optimizer = Adam(model.parameters(), lr=0.005)
criterion = nn.BCELoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)


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


#* save output predictions
eos = EpochOutputStore()
eos.attach(train_evaluator, 'output_total')


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


total_loss = []
acc = []
#* print accuracy & average loss every epoch
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics

    total_loss.append(metrics['loss'])
    acc.append(metrics['accuracy'])
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)


#! Train the model
# trainer.run(train_loader, max_epochs=100)


# plot_data()
# output = train_evaluator.state.output_total
# output = np.array([tup[0].tolist() for tup in output]).reshape(-1)
# output = np.round(output)

# plot_data(X, output)




#! Plotting
#* plot loss evolution
def plot_loss(total_loss, acc):

    ax = plt.gca()
    ax.plot(total_loss, label='Loss')
    ax.plot(acc, label='Accuracy')

    plt.ylabel('Loss/Accuracy')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Loss reduction over time')
    plt.legend()
    plt.show()

# plot_loss(total_loss, acc)


# TODO: fix contours
# output = train_evaluator.state.output_total
# print(output)
# output = np.array([tup[0].tolist() for tup in output]).reshape(-1)

# def plot_decision_boundary(X, y):
#     fig = plt.figure(figsize=(10,6))
#     ax = plt.gca()
#     ax.scatter(X[:, 0], X[:, 1], c=y ,cmap='viridis', s=30, zorder=3)
#     ax.axis('tight')
#     ax.axis('on')
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
#                          np.linspace(*ylim, num=200))
#     n_classes = len(np.unique(y))
#     contours = ax.contourf(xx, yy, output, alpha=0.3,
#                            levels=np.arange(n_classes + 1) - 0.5,
#                            cmap='viridis',
#                            zorder=1)

#     ax.set(xlim=xlim, ylim=ylim)
#     plt.show()


# plot_decision_boundary(X, y)