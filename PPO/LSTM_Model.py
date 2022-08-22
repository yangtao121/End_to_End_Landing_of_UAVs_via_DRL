import tensorflow as tf
from tensorflow.keras import Model


class Model1(Model):
    def __init__(self, width, height):
        super(Model1, self).__init__()
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

        self.IMG_Q_1 = tf.keras.layers.Dense(32, activation='relu', name="IMG_Q_1")

        self.IMG_Q_2 = tf.keras.layers.Dense(32, activation='relu', name="IMG_Q_2")

        self.IMG_Q = tf.keras.layers.Dense(18, activation='relu', name="IMG_Q")

        self.IMG_K_1 = tf.keras.layers.Dense(32, activation='relu', name="IMG_K_1")

        self.IMG_K_2 = tf.keras.layers.Dense(32, activation='relu', name="IMG_K_2")

        self.IMG_K = tf.keras.layers.Dense(18, activation='relu', name="IMG_K")

        self.IMG_V_1 = tf.keras.layers.Dense(32, activation='relu', name="IMG_V_1")

        self.IMG_V_2 = tf.keras.layers.Dense(32, activation='relu', name="IMG_V_2")

        self.IMG_V = tf.keras.layers.Dense(18, activation='relu', name="IMG_V")
        # 时间序列状态处理网络
        self.state_input = tf.keras.layers.InputLayer(input_shape=(18,), name='state_input')

        self.state_layer = tf.keras.layers.LSTM(64)

        self.state_Q_1 = tf.keras.layers.Dense(32, activation='relu', name="state_Q_1")
        self.state_Q = tf.keras.layers.Dense(18, activation='relu', name="state_Q")

        self.state_K_1 = tf.keras.layers.Dense(32, activation='relu', name="state_K_1")
        self.state_K = tf.keras.layers.Dense(18, activation='relu', name="state_K")

        self.state_V_1 = tf.keras.layers.Dense(32, activation='relu', name="state_V_1")
        self.state_V = tf.keras.layers.Dense(18, activation='relu', name="state_V")

        # 拼装层
        # self.concat = tf.keras.layers.Concatenate(axis=1)

        # seq2seq:self-attention
        self.attention = tf.keras.layers.Attention(use_scale=True, name='attention')

        # 输出层
        self.output_layer1 = tf.keras.layers.Dense(64, activation='relu', name="output_layer1")

        self.vel_1 = tf.keras.layers.Dense(32, activation='relu', name="vel_1")
        self.vel_2 = tf.keras.layers.Dense(18, activation='relu', name="vel_2")
        self.vel = tf.keras.layers.Dense(3, activation='tanh', name="vel")

    @tf.function
    def call(self, IMG, state):
        # print('2')
        # nums = IMG.shape[0]
        # if nums == 1:
        #     state = tf.expand_dims(state, axis=0)
        # print('3')

        # IMG输入
        IMG = self.conv1(IMG)
        # print('4')
        IMG = self.pool1(IMG)
        # print('5')
        IMG = self.conv2(IMG)
        IMG = self.pool2(IMG)
        # print('9')
        IMG = self.flatten(IMG)
        IMG = tf.expand_dims(IMG, axis=1)
        # print('10')
        IMG = self.IMG_layer1(IMG)
        # print('11')
        IMG = self.IMG_layer2(IMG)
        # print('6')

        IMG_Q = self.IMG_Q_1(IMG)
        IMG_Q = self.IMG_Q_2(IMG_Q)
        IMG_Q = self.IMG_Q(IMG_Q)

        IMG_K = self.IMG_K_1(IMG)
        IMG_K = self.IMG_K_2(IMG_K)
        IMG_K = self.IMG_K(IMG_K)

        IMG_V = self.IMG_V_1(IMG)
        IMG_V = self.IMG_V_2(IMG_V)
        IMG_V = self.IMG_V(IMG_V)
        # print('7')

        # state输入
        state = self.state_input(state)
        state = self.state_layer(state)
        state = tf.expand_dims(state, axis=1)

        state_Q = self.state_Q_1(state)
        state_Q = self.state_Q(state_Q)

        state_V = self.state_V_1(state)
        state_V = self.state_V(state_V)

        state_K = self.state_K_1(state)
        state_K = self.state_K(state_K)

        Q = tf.concat([IMG_Q, state_Q], axis=1)
        K = tf.concat([IMG_K, state_K], axis=1)
        V = tf.concat([IMG_V, state_V], axis=1)

        # print('8')

        # seq2seq
        attention = self.attention(inputs=[Q, V, K])
        attention = self.flatten(attention)

        # 输出
        out = self.output_layer1(attention)

        vel = self.vel_1(out)
        vel = self.vel_2(vel)
        vel = self.vel(vel)

        output = tf.concat([vel], axis=1)
        return output


class V_Model(Model):
    def __init__(self, width, height):
        super(V_Model, self).__init__()
        self.width = width
        self.height = height
        # 图像处理网络
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=4,
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

        self.IMG_Q_1 = tf.keras.layers.Dense(32, activation='relu', name="IMG_Q_1")

        self.IMG_Q_2 = tf.keras.layers.Dense(32, activation='relu', name="IMG_Q_2")

        self.IMG_Q = tf.keras.layers.Dense(18, activation='relu', name="IMG_Q")

        self.IMG_K_1 = tf.keras.layers.Dense(32, activation='relu', name="IMG_K_1")

        self.IMG_K_2 = tf.keras.layers.Dense(32, activation='relu', name="IMG_K_2")

        self.IMG_K = tf.keras.layers.Dense(18, activation='relu', name="IMG_K")

        self.IMG_V_1 = tf.keras.layers.Dense(32, activation='relu', name="IMG_V_1")

        self.IMG_V_2 = tf.keras.layers.Dense(32, activation='relu', name="IMG_V_2")

        self.IMG_V = tf.keras.layers.Dense(18, activation='relu', name="IMG_V")
        # 时间序列状态处理网络
        self.state_input = tf.keras.layers.InputLayer(input_shape=(18,), name='state_input')

        self.state_layer = tf.keras.layers.LSTM(64)

        self.state_Q_1 = tf.keras.layers.Dense(32, activation='relu', name="state_Q_1")
        self.state_Q = tf.keras.layers.Dense(18, activation='relu', name="state_Q")

        self.state_K_1 = tf.keras.layers.Dense(32, activation='relu', name="state_K_1")
        self.state_K = tf.keras.layers.Dense(18, activation='relu', name="state_K")

        self.state_V_1 = tf.keras.layers.Dense(32, activation='relu', name="state_V_1")
        self.state_V = tf.keras.layers.Dense(18, activation='relu', name="state_V")

        # 拼装层
        # self.concat = tf.keras.layers.Concatenate(axis=1)

        # seq2seq:self-attention
        self.attention = tf.keras.layers.Attention(use_scale=True, name='attention')

        # 输出层
        self.output_layer1 = tf.keras.layers.Dense(64, activation='relu', name="output_layer1")

        self.value_1 = tf.keras.layers.Dense(32, activation='relu', name="value_1")
        self.value_2 = tf.keras.layers.Dense(18, activation='relu', name="value_2")
        self.value = tf.keras.layers.Dense(1, name="value")

    @tf.function
    def call(self, IMG, state):
        # print('2')
        # nums = IMG.shape[0]
        # if nums == 1:
        #     state = tf.expand_dims(state, axis=0)
        # print('3')

        # IMG输入
        IMG = self.conv1(IMG)
        # print('4')
        IMG = self.pool1(IMG)
        # print('5')
        IMG = self.conv2(IMG)
        IMG = self.pool2(IMG)
        # print('9')
        IMG = self.flatten(IMG)
        IMG = tf.expand_dims(IMG, axis=1)
        # print('10')
        IMG = self.IMG_layer1(IMG)
        # print('11')
        IMG = self.IMG_layer2(IMG)
        # print('6')

        IMG_Q = self.IMG_Q_1(IMG)
        IMG_Q = self.IMG_Q_2(IMG_Q)
        IMG_Q = self.IMG_Q(IMG_Q)

        IMG_K = self.IMG_K_1(IMG)
        IMG_K = self.IMG_K_2(IMG_K)
        IMG_K = self.IMG_K(IMG_K)

        IMG_V = self.IMG_V_1(IMG)
        IMG_V = self.IMG_V_2(IMG_V)
        IMG_V = self.IMG_V(IMG_V)
        # print('7')

        # state输入
        state = self.state_input(state)
        state = self.state_layer(state)
        state = tf.expand_dims(state, axis=1)

        state_Q = self.state_Q_1(state)
        state_Q = self.state_Q(state_Q)

        state_V = self.state_V_1(state)
        state_V = self.state_V(state_V)

        state_K = self.state_K_1(state)
        state_K = self.state_K(state_K)

        Q = tf.concat([IMG_Q, state_Q], axis=1)
        K = tf.concat([IMG_K, state_K], axis=1)
        V = tf.concat([IMG_V, state_V], axis=1)

        # print('8')

        # seq2seq
        attention = self.attention(inputs=[Q, V, K])
        attention = self.flatten(attention)

        # 输出
        out = self.output_layer1(attention)

        v = self.value_1(out)
        v = self.value_2(v)
        v = self.value(v)

        return v


if __name__ == "__main__":
    import numpy as np

    model = V_Model(64, 64)

    IMG = np.random.normal(size=(2, 64, 64, 1))
    state = np.random.normal(size=(2, 5, 18))
    # state = np.array([state, state])
    # print(state.shape)

    mu = model(IMG, state)
    print(mu)
