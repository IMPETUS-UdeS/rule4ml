import keras
import tensorflow as tf
from packaging import version

# Check Keras version
if version.parse(keras.__version__).major >= 3:
    from keras import ops as kops
else:
    kops = None


def parametric_mape(y_index, name="", eps=1e-6):
    def mape(y_true, y_pred):
        y_true = y_true[..., y_index]
        y_pred = y_pred[..., y_index]

        if kops is None:
            diff = tf.abs(y_true - y_pred) / tf.maximum(tf.abs(y_true), eps)
            result = 100.0 * tf.reduce_mean(diff, axis=-1)
        else:
            diff = tf.abs(y_true - y_pred) / tf.maximum(tf.abs(y_true), eps)
            result = 100.0 * kops.mean(diff, axis=-1)

        return result

    if name == "":
        name = y_index
    mape.__name__ = f"mape_{name}"

    return mape


def parametric_smape(y_index, name="", eps=1e-6):
    def smape(y_true, y_pred):
        y_true = y_true[..., y_index]
        y_pred = y_pred[..., y_index]

        if kops is None:
            diff = tf.abs(y_true - y_pred) / tf.maximum(tf.abs(y_true) + tf.abs(y_pred), eps)
            result = 100.0 * tf.reduce_mean(diff, axis=-1)
        else:
            diff = kops.absolute(y_true - y_pred) / tf.maximum(
                kops.absolute(y_true) + kops.absolute(y_pred), eps
            )
            result = 100.0 * kops.mean(diff, axis=-1)

        return result

    if name == "":
        name = y_index
    smape.__name__ = f"smape_{name}"

    return smape


def parametric_r2(y_index, name="", eps=1e-6):
    def r2_score(y_true, y_pred):
        y_true = y_true[..., y_index]
        y_pred = y_pred[..., y_index]

        if kops is None:
            mean_true = tf.reduce_mean(y_true, axis=-1, keepdims=True)
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
            ss_tot = tf.reduce_sum(tf.square(y_true - mean_true), axis=-1)
        else:
            mean_true = kops.mean(y_true, axis=-1, keepdims=True)
            ss_res = kops.sum(kops.square(y_true - y_pred), axis=-1)
            ss_tot = kops.sum(kops.square(y_true - mean_true), axis=-1)

        result = 1.0 - ss_res / (ss_tot + eps)
        return result

    if name == "":
        name = y_index
    r2_score.__name__ = f"r2_{name}"

    return r2_score


if __name__ == "__main__":
    import numpy as np

    print(f"Keras {version.parse(keras.__version__)}")

    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(16, activation="relu")(inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(5, activation="relu")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="mae", metrics=[parametric_smape(0), parametric_smape(1)])

    x_test = np.random.uniform(size=[10, 10])
    y_test = np.random.uniform(size=[10, 5])

    model.evaluate(x_test, y_test)
