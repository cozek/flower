import utils
from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.array([[0., 0.], [1., 1.]])
y = np.array([0, 1])

# # Ensure y is 2D
# if y.ndim == 1:
#     y = y.reshape((-1, 1))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1, 
                    warm_start=True)

clf.fit(X, y)

# print(clf.coefs_)
# print(len(clf.coefs_))
for a in clf.coefs_:
    print(a.shape)
print(clf.n_layers_)
print(clf.get_params())

clf_2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(5, 2), random_state=1,
                      warm_start=True)

clf_2.coefs_=clf.coefs_.copy()
clf_2.intercepts_ = clf.intercepts_.copy()

hidden_layer_sizes = list(clf.get_params()['hidden_layer_sizes'])

layer_units = [0] + hidden_layer_sizes + [0]

clf_2.n_layers_ = len(layer_units)
clf_2.out_activation_ = clf.out_activation_

clf_2.fit(X, y)

print(clf.coefs_)
print(clf_2.coefs_)
