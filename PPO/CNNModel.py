import tensorflow as tf
from tensorflow.keras import Model


class CNNModel1(Model):

    def __init__(self, width, height):
        super(CNNModel1, self).__init__()
        # self.inputs = tf.keras.layers.InputLayer(input_shape=(width, height, 1))
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=4,
            input_shape=(width, height, 1),
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=4,
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        )
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(
            128, activation='tanh'
        )
        self.layer2 = tf.keras.layers.Dense(
            64, activation="tanh"
        )

        self.pos = tf.keras.layers.Dense(
            3, activation="tanh"
        )

        self.vel = tf.keras.layers.Dense(3, activation="tanh")
        self.rot_x = tf.keras.layers.Dense(3, activation="softmax")
        self.rot_y = tf.keras.layers.Dense(3, activation="softmax")
        self.rot_z = tf.keras.layers.Dense(3, activation="softmax")
        self.omega = tf.keras.layers.Dense(3, activation="tanh")

        # self.flatten2 = tf.keras.layers.Flatten()

    def call(self, x):
        # x = self.inputs(x)
        nums = x.shape[0]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        pos = self.pos(x)
        vel = self.vel(x)
        rot_x = self.rot_x(x)
        rot_y = self.rot_y(x)
        rot_z = self.rot_z(x)
        omega = self.omega(x)
        out = tf.concat([pos, vel, rot_x, rot_y, rot_z, omega], axis=0)
        out = tf.reshape(out, (nums, 18))
        # out = tf.squeeze(out,axis=0)

        return out


class V_CNNModel1(Model):

    def __init__(self, width, height):
        super(V_CNNModel1, self).__init__()
        # self.inputs = tf.keras.layers.InputLayer(input_shape=(width, height, 1))
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=4,
            input_shape=(width, height, 1),
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=4,
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        )
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(
            64, activation='tanh'
        )
        self.layer2 = tf.keras.layers.Dense(
            64, activation="tanh"
        )

        self.out = tf.keras.layers.Dense(1)

        # self.flatten2 = tf.keras.layers.Flatten()

    def call(self, x):
        # x = self.inputs(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out(x)
        # out = tf.squeeze(out,axis=0)

        return x


def CNNModel(width, height):
    # inputs = tf.keras.layers.InputLayer(input_shape=(width, height, 1))
    conv1 = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=4,
        input_shape=(width, height, 1),
        activation='relu'
    )
    pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv1)
    conv2 = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=4,
        activation='relu'
    )(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(conv2)
    flatten = tf.keras.layers.Flatten()(pool2)
    layer1 = tf.keras.layers.Dense(
        128, activation='tanh'
    )(flatten)
    layer2 = tf.keras.layers.Dense(
        64, activation="tanh"
    )(layer1)

    pos = tf.keras.layers.Dense(
        3, activation="tanh"
    )(layer2)
    vel = tf.keras.layers.Dense(
        3, activation="tanh"
    )(layer2)
    rot_x = tf.keras.layers.Dense(3, activation="softmax")(layer2)
    rot_y = tf.keras.layers.Dense(3, activation="softmax")(layer2)
    rot_z = tf.keras.layers.Dense(3, activation="softmax")(layer2)
    omega = tf.keras.layers.Dense(3, activation="tanh")(layer2)
    model = tf.keras.Model(
        inputs=conv1,
        outputs=[pos, vel, rot_x, rot_y, rot_z, omega]
    )
    return model


if __name__ == "__main__":
    import numpy as np

    model = CNNModel1(32, 32)
    model.summary()
    inputs = np.random.normal(size=(2, 32, 32, 1))
    a = model(inputs)
    print(a)
    tf.keras.utils.plot_model(model, 'policy.png', show_shapes=True)
