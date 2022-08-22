import tensorflow as tf
from RL.common.PolicyBase import policy_base


class critic_policy(policy_base):
    def __init__(self, Model=None, file_name=None):
        super().__init__(Model=Model, file_name=file_name)

    @tf.function
    def get_value(self, IMG):
        v = self.Model(IMG)
        v = tf.squeeze(v)
        return v

    def get_variable(self):
        return self.Model.trainable_variables

    def __call__(self, IMG):
        return self.Model(IMG)
