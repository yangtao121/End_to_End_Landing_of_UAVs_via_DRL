import tensorflow as tf
import tensorflow_probability as tfp
from RL.common.PolicyBase import policy_base


class gaussian_policy(policy_base):
    def __init__(self, Model=None, file_name=None):
        super().__init__(Model=Model, file_name=file_name)
        self.log_std = tf.Variable(tf.zeros((3,))-0.5, trainable=True)
        self.mu = tf.zeros(shape=(3,), dtype=tf.float32)
        self.sigma = tf.ones(shape=(3,), dtype=tf.float32)

    # @tf.function
    # def get_action(self, IMG):
    #     # print('1')
    #     mu = self.Model(IMG)
    #     mu = tf.squeeze(mu, axis=0)
    #     # print(mu)
    #     # print('2')
    #     sigma = tf.exp(self.log_std)
    #
    #     # dist = tfp.distributions.Normal(mu, sigma)
    #     # action = tf.squeeze(dist.sample(), axis=0)
    #     #
    #     # prob = tf.squeeze(dist.prob(action), axis=0)
    #
    #     return mu, 1
    #
    # @tf.function
    def get_action(self, IMG):
        a_adv = self.Model(IMG)
        a_adv = tf.squeeze(a_adv, axis=0)
        a_std = tf.exp(self.log_std)
        # print(a_adv)

        noise_dist = tfp.distributions.Normal(self.mu, self.sigma)
        noise = noise_dist.sample()
        # print(noise)
        action = a_adv + noise * a_std

        prob = noise_dist.prob(noise)

        return action, prob

    # def get_action(self, IMG):
    #     a_adv = self.Model(IMG)
    #     a_adv = tf.squeeze(a_adv, axis=0)
    #     a_std = tf.exp(self.log_std)
    #     dist = tfp.distributions.Normal(a_adv, a_std)
    #
    #     # noise_dist = tfp.distributions.Normal(self.mu, self.sigma)
    #     # noise = noise_dist.sample()
    #     action = dist.sample()
    #
    #     prob = dist.prob(action)
    #
    #     return action, prob

    def get_std(self):
        return self.log_std.numpy()

    def init_log_std(self):
        self.log_std = tf.Variable(tf.zeros((3,)) - 0.5, trainable=True)

    def __call__(self, IMG):
        mu = self.Model(IMG)
        sigma = tf.exp(self.log_std)

        return mu, sigma

    def get_variable(self):
        return self.Model.trainable_variables + [self.log_std]

    def set_std(self, data):
        self.log_std = tf.Variable(data, dtype=tf.float32)


class gaussian_policy_test(policy_base):
    def __init__(self, Model=None, file_name=None):
        super().__init__(Model=Model, file_name=file_name)
        self.log_std = tf.Variable(tf.zeros((3,)) - 0.5, trainable=True)
        self.mu = tf.zeros(shape=(3,), dtype=tf.float32)
        self.sigma = tf.ones(shape=(3,), dtype=tf.float32)

    # @tf.function
    def get_action(self, IMG):
        # print('1')
        mu = self.Model(IMG)
        mu = tf.squeeze(mu, axis=0)
        print(mu.numpy())
        # print('2')
        # sigma = tf.exp(self.log_std)

        # dist = tfp.distributions.Normal(mu, sigma)
        # action = tf.squeeze(dist.sample(), axis=0)
        #
        # prob = tf.squeeze(dist.prob(action), axis=0)

        return mu, 1

    def __call__(self, IMG):
        mu = self.Model(IMG)
        sigma = tf.exp(self.log_std)

        return mu, sigma

    def get_variable(self):
        return self.Model.trainable_variables + [self.log_std]

    def get_std(self):
        return self.log_std.numpy()

    def set_std(self, data):
        self.log_std = tf.Variable(data)
