import warnings
import utils
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import flwr as fl

def get_model_parameters(model):
    return [
        model.coefs_,
        model.intercepts_
    ]

def set_model_params(model, params, config):
    model.coefs_= params.coefs_.copy()
    model.intercepts_ = params.intercepts_.copy()

    hidden_layer_sizes = config['hidden_layer_sizes']
    layer_units = [0] + hidden_layer_sizes + [0]
    
    model.n_layers_ = len(layer_units)
    model.out_activation_ = params.out_activation_
    # clf_2.set_params(**clf.get_params())

    

class MnistNNClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            super().__init__()
            self.model = None
        
        def get_parameters(self, config):  # type: ignore
            # create a random model
            if self.model is None:
                self.model = MLPClassifier(**config['initial_model_config'])
            return get_model_parameters(self.model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters, config)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

if __name__ == '__main__':

    (X, y), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X, y) = utils.partition(X, y, 10)[partition_id]



# X = np.array([[0., 0.], [1., 1.]])
# y = np.array([0, 1])

# # Ensure y is 2D
# if y.ndim == 1:
#     y = y.reshape((-1, 1))


model = MLPClassifier(warm_start=True)

# clf.fit(X, y)

# # print(clf.coefs_)
# # print(len(clf.coefs_))
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