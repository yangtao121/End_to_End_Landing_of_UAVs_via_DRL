import tensorflow as tf
from tensorflow.keras import Model


class Model1(Model):
    def __init__(self, width, height):
        super(Model1, self).__init__()
        self.width = width
        self.height = height
        # 图像处理网络
        initializer = tf.keras.initializers.orthogonal()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=7,
            input_shape=(width, height, 1),
            activation='relu',
            name='conv1',
            kernel_initializer=initializer
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu', name="conv2",
                                            kernel_initializer=initializer)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2")
        self.flatten = tf.keras.layers.Flatten()
        self.IMG_layer1 = tf.keras.layers.Dense(128, activation='tanh', name="IMG_layer1",
                                                kernel_initializer=initializer)
        self.IMG_layer2 = tf.keras.layers.Dense(64, activation="tanh", name="IMG_layer2",
                                                kernel_initializer=initializer)

        # 输出层
        self.output_layer1 = tf.keras.layers.Dense(64, activation='tanh', name="output_layer1",
                                                   kernel_initializer=initializer)

        self.vel_1 = tf.keras.layers.Dense(32, activation='tanh', name="vel_1", kernel_initializer=initializer)
        self.vel_2 = tf.keras.layers.Dense(18, activation='tanh', name="vel_2", kernel_initializer=initializer)
        self.vel = tf.keras.layers.Dense(3, activation='tanh', name="vel", kernel_initializer=initializer)

    # @tf.function
    def call(self, IMG):
        IMG = self.conv1(IMG)
        IMG = self.pool1(IMG)
        IMG = self.conv2(IMG)
        IMG = self.pool2(IMG)
        IMG = self.flatten(IMG)
        # IMG = tf.expand_dims(IMG, axis=1)
        IMG = self.IMG_layer1(IMG)
        IMG = self.IMG_layer2(IMG)
        out = self.output_layer1(IMG)
        vel = self.vel_1(out)
        vel = self.vel_2(vel)
        vel = self.vel(vel)
        output = tf.concat([vel], axis=1)
        return output


class Model2(Model):
    def __init__(self, width, height):
        super(Model2, self).__init__()
        self.width = width
        self.height = height
        # 图像处理网络
        initializer = tf.keras.initializers.zeros()
        initializer2 = tf.keras.initializers.RandomUniform(minval=-0.8, maxval=-0.2)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=7,
            input_shape=(width, height, 1),
            activation='relu',
            name='conv1',
            kernel_initializer=initializer
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu', name="conv2",
                                            kernel_initializer=initializer)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2")
        self.flatten = tf.keras.layers.Flatten()
        self.IMG_layer1 = tf.keras.layers.Dense(128, activation='tanh', name="IMG_layer1",
                                                kernel_initializer=initializer)
        self.IMG_layer2 = tf.keras.layers.Dense(64, activation="tanh", name="IMG_layer2",
                                                kernel_initializer=initializer)

        # 输出层
        self.output_layer1 = tf.keras.layers.Dense(64, activation='tanh', name="output_layer1",
                                                   kernel_initializer=initializer)

        self.velx_1 = tf.keras.layers.Dense(32, activation='tanh', name="velx_1", kernel_initializer=initializer)
        self.velx_2 = tf.keras.layers.Dense(18, activation='tanh', name="velx_2", kernel_initializer=initializer)
        self.velx = tf.keras.layers.Dense(1, activation='tanh', name="vel", kernel_initializer=initializer)

        self.vely_1 = tf.keras.layers.Dense(32, activation='tanh', name="vely_1", kernel_initializer=initializer)
        self.vely_2 = tf.keras.layers.Dense(18, activation='tanh', name="vely_2", kernel_initializer=initializer)
        self.vely = tf.keras.layers.Dense(1, activation='tanh', name="vely", kernel_initializer=initializer)

        self.velz_1 = tf.keras.layers.Dense(32, activation='tanh', name="velz_1", kernel_initializer=initializer2)
        self.velz_2 = tf.keras.layers.Dense(18, activation='tanh', name="velz_2", kernel_initializer=initializer2)
        self.velz = tf.keras.layers.Dense(1, activation='tanh', name="velz", kernel_initializer=initializer2)

    # @tf.function
    def call(self, IMG):
        IMG = self.conv1(IMG)
        IMG = self.pool1(IMG)
        IMG = self.conv2(IMG)
        IMG = self.pool2(IMG)
        IMG = self.flatten(IMG)
        # IMG = tf.expand_dims(IMG, axis=1)
        IMG = self.IMG_layer1(IMG)
        IMG = self.IMG_layer2(IMG)
        out = self.output_layer1(IMG)
        velx = self.velx_1(out)
        velx = self.velx_2(velx)
        velx = self.velx(velx)

        vely = self.vely_1(out)
        vely = self.vely_2(vely)
        vely = self.vely(vely)

        velz = self.velz_1(out)
        velz = self.velz_2(velz)
        velz = self.velz(velz)

        output = tf.concat([velx, vely, velz], axis=1)
        return output


class V_Model(Model):
    def __init__(self, width, height):
        super(V_Model, self).__init__()
        self.width = width
        self.height = height
        # 图像处理网络
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=7,
            input_shape=(width, height, 1),
            activation='relu',
            name='conv1'
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu', name="conv2")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2")
        self.flatten = tf.keras.layers.Flatten()
        self.IMG_layer1 = tf.keras.layers.Dense(128, activation='tanh', name="IMG_layer1")
        self.IMG_layer2 = tf.keras.layers.Dense(64, activation="tanh", name="IMG_layer2")

        # 输出层
        self.output_layer1 = tf.keras.layers.Dense(64, activation='relu', name="output_layer1")

        self.value_1 = tf.keras.layers.Dense(32, activation='relu', name="value_1")
        self.value_2 = tf.keras.layers.Dense(18, activation='relu', name="value_2")
        self.value = tf.keras.layers.Dense(1, name="value")

    @tf.function
    def call(self, IMG):
        IMG = self.conv1(IMG)
        IMG = self.pool1(IMG)
        IMG = self.conv2(IMG)
        IMG = self.pool2(IMG)
        IMG = self.flatten(IMG)
        # IMG = tf.expand_dims(IMG, axis=1)
        IMG = self.IMG_layer1(IMG)
        IMG = self.IMG_layer2(IMG)

        out = self.output_layer1(IMG)

        v = self.value_1(out)
        v = self.value_2(v)
        v = self.value(v)
        return v


if __name__ == "__main__":
    import numpy as np
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    model = Model1(64, 64)

    IMG = np.random.normal(size=(1, 64, 64, 1))
    # state = np.random.normal(size=(2, 5, 18))
    # state = np.array([state, state])
    # print(state.shape)

    mu = model(IMG)
    print(mu)
