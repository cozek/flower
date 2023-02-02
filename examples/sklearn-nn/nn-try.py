import warnings
import utils
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import flwr as fl

(X_train, y_train), (X_test, y_test) = utils.load_mnist()
MODEL_CONFIG = {'hidden_layer_sizes': (50,), 'warm_start': True}

def initialize_model():
    model = MLPClassifier(**MODEL_CONFIG)
    # Fitting model to random data to initialize it
    X = np.random.rand(10, 784)
    y = np.arange(10)
    model.partial_fit(X,y,y)
    return model

def get_model_parameters(model):
    return model.coefs_ + model.intercepts_

def set_model_params(model, params):
    n = len(MODEL_CONFIG['hidden_layer_sizes'])
    model.coefs_ = params[:n + 1]
    model.intercepts_ = params[n + 1:]

model = initialize_model()
# model.fit(X_train, y_train)
set_model_params(model, get_model_parameters(model))
loss = log_loss(y_test, model.predict_proba(X_test))
accuracy = model.score(X_test, y_test)
print( loss, {"accuracy": accuracy})

print(X_train.shape)

# Split train set into 10 partitions and randomly use one for training.
# partition_id = np.random.choice(10)
# (X, y) = utils.partition(X, y, 10)[partition_id]

# print(X.shape, y.shape)

# X = np.array([[0], [1]])
# y = np.array([0, 1])

# X = np.random.rand(10,784)
# y = np.arange(10)


# # Ensure y is 2D
# if y.ndim == 1:4
#     y = y.reshape((-1, 1))


# model = MLPClassifier(hidden_layer_sizes=(1,1),warm_start=True)

# model.fit(X, y)

# params = model.coefs_ + model.intercepts_

# print(model.coefs_)
# print(model.intercepts_)

# print(len(model.coefs_))
# print(len(model.intercepts_))

# print(params[:3])
# print(params[3:])

# model.set_params(**{})
# print(model.get_params()['hidden_layer_sizes'])
# hidden_layer_sizes = model.get_params()['hidden_layer_sizes']
# # layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]
# layer_units = [X.shape[1]] + list(hidden_layer_sizes) + [0]

# model._initialize(y.reshape((-1, 1)), layer_units, X.dtype)

# model.partial_fit(X, y, np.unique(y_test))


# print(np.unique(y))
# print(np.unique(y_test))


# params = model.coefs_ + model.intercepts_
# print(len(model.coefs_))
# print(len(model.intercepts_))

# print(len(params))

# print(model.coefs_)
# print(len(model.coefs_))

# print(model.get_params())
# print(model.out_activation_)


# for a in clf.coefs_:
#     print(a.shape)
# print(clf.n_layers_)
# print(clf.get_params())
# print('acc', clf.score(X_test, y_test))

# config = {'solver':'lbfgs', 'alpha':1e-5,
#                       'hidden_layer_sizes':(5, 2), 'random_state':1,
#                       'warm_start':True, 'max_iter':100}

# clf_2 = MLPClassifier(**config)

# clf_2.coefs_=clf.coefs_.copy()
# clf_2.intercepts_ = clf.intercepts_.copy()

# hidden_layer_sizes = list(clf.get_params()['hidden_layer_sizes'])

# layer_units = [0] + hidden_layer_sizes + [0]

# clf_2.n_layers_ = len(layer_units)
# clf_2.out_activation_ = clf.out_activation_
# # clf_2.set_params(**clf.get_params())
# clf_2.fit(X_test, y_test)

# print(clf.coefs_)
# print(clf_2.coefs_)
# print('acc', clf_2.score(X, y))

# print(clf.intercepts_)
# print(clf_2.intercepts_)


# model = LogisticRegression(
#     penalty="l2",
#     max_iter=10000,  # local epoch
#     warm_start=True,  # prevent refreshing weights when fitting
# )

# Setting initial parameters, akin to model.compile for keras models
# utils.set_initial_params(model)

# model.fit(X, y)
# p= utils.get_model_parameters(model)
# print(p)
# print(model.coef_)