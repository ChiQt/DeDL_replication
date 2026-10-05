"""Structured TensorFlow network adapted from the authors' sig_dnn builder.

Source: main.ipynb, cell 12, commit 054ec6ae3541474a189203a89fc34698432735f7.
See THIRD_PARTY.md for attribution.
"""
import numpy as np


def build_network(dimension):
    """Keep the author's two branches and scaled-sigmoid response layer."""
    from tensorflow.keras import Input, Model, initializers, layers

    features = Input(shape=(dimension,), name="feature_input")
    treatment = Input(shape=(3,), name="treatment_input")

    # Heterogeneous intercept a(x).
    a = layers.Dense(20, activation="relu")(features)
    a = layers.Dense(20, activation="relu")(a)
    a = layers.Dense(20, activation="relu")(a)
    a = layers.Dense(1, name="parameter_a")(a)

    # Heterogeneous treatment coefficients b(x).
    b = layers.Dense(20, activation="relu")(features)
    b = layers.Dense(20, activation="relu")(b)
    b = layers.Dense(20, activation="relu")(b)
    b = layers.Dense(3, name="parameter_b")(b)

    bt = layers.Dot(axes=1)([b, treatment])
    index = layers.Add(name="u")([a, bt])
    # The fixed unit weight reproduces the author's sigmoid layer exactly.
    sigmoid = layers.Dense(
        1, use_bias=False, trainable=False, activation="sigmoid",
        kernel_initializer=initializers.Constant(1), name="sigmoid_function"
    )(index)
    response = layers.Dense(1, use_bias=False, name="output_sigmoid")(sigmoid)
    return Model(inputs=[features, treatment], outputs=response)


class TensorFlowModel:
    """Train the author's structured network with the common initialization."""

    def __init__(self, dimension, weights, initial_scale, learning_rate):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        self.net = build_network(dimension)

        def branch_layers(output_name):
            result = []
            layer = self.net.get_layer(output_name)
            while isinstance(layer, tf.keras.layers.Dense):
                result.append(layer)
                layer = layer.input._keras_history.layer
            return list(reversed(result))

        layers = branch_layers("parameter_a") + branch_layers("parameter_b")
        for layer, weight_pair in zip(layers, weights):
            layer.set_weights(list(weight_pair))
        self.net.get_layer("output_sigmoid").set_weights([
            np.array([[initial_scale]], dtype=np.float32)
        ])
        self.parameter_model = tf.keras.Model(self.net.inputs, [
            self.net.get_layer("parameter_a").output,
            self.net.get_layer("parameter_b").output
        ])
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-7
        )
        self.step = tf.function(self._step, input_signature=[
            tf.TensorSpec([None, dimension], tf.float32),
            tf.TensorSpec([None, 3], tf.float32), tf.TensorSpec([None], tf.float32)
        ])

    def bind_data(self, x, treatment, y):
        self.x, self.treatment, self.y = x, treatment, y

    def _step(self, x, treatment, y):
        with self.tf.GradientTape() as tape:
            predictions = self.tf.reshape(self.net([x, treatment], training=True), [-1])
            loss = self.tf.reduce_mean(self.tf.square(predictions - y))
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
        return loss

    def train_epoch(self, order, batch_size):
        for start in range(0, len(order), batch_size):
            users = order[start:start + batch_size]
            self.step(self.x[users], self.treatment[users], self.y[users])

    def predict(self, x, treatment):
        return self.net([x, treatment], training=False).numpy().reshape(-1)

    def parameters(self, x):
        a, b = self.parameter_model([x, np.zeros((len(x), 3), np.float32)], training=False)
        scale = float(self.net.get_layer("output_sigmoid").get_weights()[0][0, 0])
        return np.column_stack([a.numpy(), b.numpy(), np.full(len(x), scale)]).astype(np.float64)

    def save(self, path):
        self.net.save(str(path) + ".h5", include_optimizer=False)

    def versions(self):
        return {"tensorflow": self.tf.__version__, "numpy": np.__version__}
