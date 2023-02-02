from sklearn.neural_network import MLPClassifier
import utils
import numpy as np
from functools import reduce

MODEL_CONFIG = {'hidden_layer_sizes': (50,), 'warm_start': True, 'max_iter': 1, 'learning_rate_init':0.0001}

def initialize_model():
    model = MLPClassifier(**MODEL_CONFIG)
    # Fitting model to random data to initialize it
    X = np.random.rand(10, 784)
    y = np.arange(10)
    model.partial_fit(X, y, y)
    return model

def get_model_parameters(model):
    return model.coefs_ + model.intercepts_

def set_model_params(model, params):
    n = len(MODEL_CONFIG['hidden_layer_sizes'])
    model.coefs_ = params[:n + 1]
    model.intercepts_ = params[n + 1:]

def aggregate(results):
    """Compute weighted average."""
    # https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/aggregate.py
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

models = [initialize_model() for i in range(10)]
server_model = initialize_model()

n_rounds = 1

(X_train, y_train), (X_test, y_test) = utils.load_mnist()

for i in range(n_rounds):
    print(f'Round {i}')
    weights = []
    for j,m in enumerate(models):
        partition_id = np.random.choice(10)
        (X, y) = utils.partition(X_train, y_train, 10)[partition_id]
        m.fit(X,y)
        print(j,m.score(X,y))
        m_weight = get_model_parameters(m)
        for w in m_weight:
            w *= len(X)
            w /= len(X_train)
        weights.append(m_weight)

    # print(weights)
    w0 = weights[0]
    for w in range(1,len(weights)):
        for j in range(len(weights[w])):
            w0[j]+=weights[w][j]

    set_model_params(server_model, w0)
    print('Server_Acc:' ,server_model.score(X_test,y_test))