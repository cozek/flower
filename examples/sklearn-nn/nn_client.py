import warnings
import utils

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

import numpy as np
import flwr as fl

MODEL_CONFIG = {'hidden_layer_sizes': (300,), 'warm_start':True}

def get_model_parameters(model):
    return model.coefs_ + model.intercepts_

def set_model_params(model, params):
    n = len(MODEL_CONFIG['hidden_layer_sizes'])
    model.coefs_= params[:n+1]
    model.intercepts_ = params[n+1:]

def initialize_model():
    model = MLPClassifier(**MODEL_CONFIG)
    # Fitting model to random data to initialize it
    X = np.random.rand(10, 784)
    y = np.arange(10)
    model.partial_fit(X,y,y)
    return model

class MnistNNClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            super().__init__()
            self.model = initialize_model()
            _ , (self.X_test, self.y_test) = utils.load_mnist()

        
        def get_parameters(self, config):  # type: ignore
            return get_model_parameters(self.model)

        def fit(self, parameters, config):  # type: ignore
            set_model_params(self.model, parameters)
            # Ignore convergence failure due to low local epochs

            (X_train, y_train), _ = utils.load_mnist()

            # Split train set into 10 partitions and randomly use one for training.
            partition_id = np.random.choice(10)
            (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return get_model_parameters(self.model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            set_model_params(self.model, parameters)
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
            accuracy = self.model.score(self.X_test, self.y_test)
            return loss, len(self.X_test), {"accuracy": accuracy}
    
if __name__ == '__main__':
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080", client=MnistNNClient())
