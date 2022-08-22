import tensorflow as tf
import tensorflow_probability as tfp
from RL.common.PolicyBase import policy_base


class gaussian_policy(policy_base):
    def __init__(self, Model=None, file_name=None):
        super().__init__(Model=Model, file_name=file_name)

    # def get_action(self, IMG, states):
    #     # print('1')
    #     mu, sigma = self.Model(IMG, states)
    #     mu = tf.squeeze(mu, axis=0)
    #     print(sigma)
    #     # print(mu)
    #     # print('2')
    #     # sigma = tf.exp(self.log_std)
    #
    #     # dist = tfp.distributions.Normal(mu, sigma)
    #     # action = tf.squeeze(dist.sample(), axis=0)
    #     #
    #     # prob = tf.squeeze(dist.prob(action), axis=0)
    #
    #     return mu, 1

    def get_action(self, IMG, states):
        # print('1')
        mu, log_sigma = self.Model(IMG, states)
        sigma = tf.exp(log_sigma)
        sigma = tf.maximum(sigma, 1e-3)
        # sigma = tf.clip_by_value(sigma, 1e-2, 1e2)
        # mu = tf.squeeze(mu, axis=0)
        # print('2')
        # sigma = tf.exp(sigma)

        dist = tfp.distributions.Normal(mu, sigma)
        action = tf.squeeze(dist.sample(), axis=0)

        prob = tf.squeeze(dist.prob(action), axis=0)
        # prob = tf.clip_by_value(prob, 0, 1)
        # print(sigma)
        # print(prob)
        # print('------------------------')

        return action, prob

    def __call__(self, IMG, states):
        mu, log_sigma = self.Model(IMG, states)
        sigma = tf.exp(log_sigma)
        sigma = tf.maximum(sigma, 1e-3)
        # sigma = tf.clip_by_value(sigma, 1e-2, 1e2)
        # sigma = tf.exp(sigma)

        return mu, sigma

    def get_variable(self):
        return self.Model.trainable_variables
