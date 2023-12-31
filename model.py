# %%
import tensorflow as tf
from tensorflow.keras import Model
from breakout import epsilon, EPSILON_DECAY


# %%
class Model(Model):
    def __init__(self, width, height, actions):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0025,
            beta_1=EPSILON_DECAY,
            beta_2=EPSILON_DECAY,
            epsilon=epsilon,
            amsgrad=False,
        )

        self.conv1 = tf.keras.layers.Conv2D(
            16, [8, 8], strides=4, input_shape=(width, height, 4), activation="relu"
        )

        self.conv2 = tf.keras.layers.Conv2D(32, [4, 4], strides=2, activation="relu")

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(actions, activation="linear")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
