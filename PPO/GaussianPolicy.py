import tensorflow as tf
import tensorflow_probability as tfp
from RL.common.PolicyBase import policy_base


class gaussian_policy(policy_base):
    def __init__(self, Model=None, file_name=None):
        super().__init__(Model=Model, file_name=file_name)
        self.log_std = tf.Variable(tf.zeros((3,))-0.5, trainable=True)

    # def get_action(self, IMG, states):
    #     # print('1')
    #     mu = self.Model(IMG, states)
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

    def get_action(self, IMG):
        # print('1')
        mu = self.Model(IMG)
        # mu = tf.squeeze(mu, axis=0)
        # print('2')
        sigma = tf.exp(self.log_std)

        dist = tfp.distributions.Normal(mu, sigma)
        action = tf.squeeze(dist.sample(), axis=0)

        prob = tf.squeeze(dist.prob(action), axis=0)

        return action, prob

    def __call__(self, IMG):
        mu = self.Model(IMG)
        sigma = tf.exp(self.log_std)

        return mu, sigma

    def get_variable(self):
        return self.Model.trainable_variables + [self.log_std]

    def get_std(self):
        return self.log_std.numpy()

    def set_std(self, data):
        self.log_std = tf.Variable(data, dtype=tf.float32)
