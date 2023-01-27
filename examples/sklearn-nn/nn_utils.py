from sklearn.neural_network import MLPClassifier

def set_model_params(model, params, config):
    n = len(MODEL_CONFIG['hidden_layer_sizes'])
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