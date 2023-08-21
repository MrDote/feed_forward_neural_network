import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVC


iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target
m=X[:, 0]
n=X[:, 1]


svm = SVC(C=0.5, kernel='linear')
svm.fit(X, y)


def plot_decision_boundary(X,y,clf=None):
    fig=plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y,cmap='viridis',s=30, zorder=3)
    ax.axis('tight')
    ax.axis('on')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='viridis',
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    plt.show()


plot_decision_boundary(X, y, svm)