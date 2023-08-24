# Feedforward Neural Network (FFNN)

Eg. for classification we want to separate out the classes by drawing a decision boundary
Input data (as given) is not linearly separable.
By performing non-linear transformation of layer's inputs able to project data from one vector space to another -> draw a complex decision boundary to separate classes (similar to SVMs)


## Binary classification (Linear regression)

- 1 output node outputting P(class=1) -> P(class=0) = 1 - P(class=1) with **sigmoid** activation func

- Loss fn: binary_crossentropy

- y: 1 column with 0/1 values


## Multiclass classification (Softmax regression)

- 1 output node for each class with **softmax** activation func (probabilities sum to 1)

- Loss fn: categorical_crossentropy

- y: one-hot

P.S multilabel == one sample can have more than one label (e.g. 0, 1, 1 in one-hot)