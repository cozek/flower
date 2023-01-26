import warnings
import utils

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

import numpy as np
import flwr as fl

MODEL_CONFIG = {'hidden_layer_sizes': (10, 2), 'max_iter': 100, 'warm_start':True}

def get_model_parameters(model):
    return model.coefs_ + model.intercepts_

def set_model_params(model, params, config):
    n = model.get_params()['hidden_layer_sizes']
    model.coefs_= params[:n+1]
    model.intercepts_ = params[n+1:]
    model.n_layers_ = n+2
    # Output for multi class "softmax"
    # Output for binary class and multi-label: "logistic"
    model.out_activation_ = 'logistic'

def initialize_model():
    model = MLPClassifier(**MODEL_CONFIG)
    hidden_layer_sizes = model.get_params()['hidden_layer_sizes']
    layer_units = [0] + hidden_layer_sizes + [0]
    model.n_layers_ = len(layer_units)
    # Output for multi class "softmax"
    # Output for binary class and multi-label: "logistic"
    model.out_activation_ = 'logistic'
    return model

class MnistNNClient(fl.client.NumPyClient):
        def __init__(self, X_train, y_train, X_test, y_test) -> None:
            super().__init__()
            self.model = None
        
        def get_parameters(self, config):  # type: ignore
            
            # server might request params and there may not be a model yet
            if self.model is None:
                self.model = initialize_model()
            return get_model_parameters(self.model)

        def fit(self, parameters, config):  # type: ignore
            if self.model is None:
                self.model = initialize_model()
            set_model_params(self.model, parameters, config)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.X_train, self.y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(self.model), len(self.X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            set_model_params(self.model, parameters, config)
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
            accuracy = self.model.score(self.X_test, self.y_test)
            return loss, len(self.X_test), {"accuracy": accuracy}