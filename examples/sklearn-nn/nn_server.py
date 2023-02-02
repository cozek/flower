import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from typing import Dict
import numpy as np

MODEL_CONFIG = {'hidden_layer_sizes': (50,), 'warm_start':True}

def get_model_parameters(model):
    return model.coefs_ + model.intercepts_

def set_model_params(model, params):
    n = len(MODEL_CONFIG['hidden_layer_sizes'])
    model.coefs_ = params[:n + 1]
    model.intercepts_ = params[n + 1:]

def initialize_model():
    model = MLPClassifier(**MODEL_CONFIG)
    # Fitting model to random data to initialize it
    X = np.random.rand(10, 784)
    y = np.arange(10)
    model.partial_fit(X, y, y)
    return model

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    config =  {"server_round": server_round, 'model_config':MODEL_CONFIG}
    return config


def get_evaluate_fn(model: MLPClassifier):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()
    X_test /= 255.

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":

    model = initialize_model()

    # utils.set_initial_params(model)
    
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )
