import warnings
import utils

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

import numpy as np
import flwr as fl


def get_model_parameters(model):
    # TODO: Find appropriate way to get model params
    return [
        model.coefs_,
        model.intercepts_
    ]

def set_model_params(model, params, config):
    # TODO: actually set model_params
    model.coefs_= params.coefs_.copy()
    model.intercepts_ = params.intercepts_.copy()

    hidden_layer_sizes = config['hidden_layer_sizes']
    layer_units = [0] + hidden_layer_sizes + [0]
    
    model.n_layers_ = len(layer_units)
    model.out_activation_ = params.out_activation_


class MnistNNClient(fl.client.NumPyClient):
        def __init__(self, X_train,y_train, X_test,y_test) -> None:
            super().__init__()
            self.model = None
        
        def get_parameters(self, config):  # type: ignore
            # create a random model
            if self.model is None:
                self.model = MLPClassifier(**config['initial_model_config'])
            return get_model_parameters(self.model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(self.model, parameters, config)
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