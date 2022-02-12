import os

import flwr as fl
import tensorflow as tf
import numpy as np

import utils

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_model():
    num_classes = 10
    input_shape = (28, 28, 1)
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def main():
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (x_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]
    (x_test, y_test) = utils.partition(X_test, y_test, 10)[partition_id]

    # reshape for convolution
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Create model
    model = create_model()
    model.compile("SGD", "sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    class TensorflowClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # fl.client.start_numpy_client("0.0.0.0:8080", client=TensorflowClient())


if __name__ == "__main__":
    main()
