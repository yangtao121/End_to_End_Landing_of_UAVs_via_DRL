import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from RL.common.functions import mkdir, standardize, gae_target
from matplotlib import pyplot as plt
from RL.common.utilit import RewardScale
import yaml


class PPO:
    def __init__(self,
                 policy,
                 critic,
                 env_args,
                 hyper_parameter,
                 worker=None,
                 net_visualize=False,
                 ):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.policy = policy
        self.critic = critic
        self.worker = worker

        self.clip_ratio = hyper_parameter.clip_ratio
        self.update_steps = hyper_parameter.update_steps
        self.tolerance = hyper_parameter.tolerance
        self.center = hyper_parameter.center
        self.reward_scale = hyper_parameter.reward_scale
        self.scale = hyper_parameter.scale
        self.clip_value = hyper_parameter.clip_value
        self.gamma = hyper_parameter.gamma
        self.lambada = hyper_parameter.lambada
        self.center_adv = hyper_parameter.center_adv

        self.hyper_parameter = hyper_parameter

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=hyper_parameter.critic_learning_rate)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=hyper_parameter.policy_learning_rate)

        self.batch_size = env_args.batch_size
        self.span = env_args.span
        self.epochs = env_args.epochs
        self.steps = env_args.steps
        self.total_steps = env_args.total_steps
        self.trajs = env_args.trajs
        self.multi_worker_num = env_args.multi_worker_num

        self.env_args = env_args

        self.mini_batch_size_num = env_args.mini_batch_size_num
        self.mini_batch_size = env_args.mini_batch_size

        if self.reward_scale:
            self.RewardScale = RewardScale(self.total_steps, self.center, self.scale)

        if net_visualize:
            self.policy.net_visual()
            self.critic.net_visual()

    # @tf.function
    def critic_train(self, IMG, target):
        if self.clip_value:
            with tf.GradientTape() as tape:
                v = self.critic(IMG)
                surrogate1 = tf.square(v[1:] - target[1:])
                surrogate2 = tf.square(
                    tf.clip_by_value(v[1:], v[:-1] - self.clip_ratio, v[:-1] + self.clip_ratio) - target[1:])
                critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        else:
            with tf.GradientTape() as tape:
                v = self.critic(IMG)
                critic_loss = tf.reduce_mean(tf.square(target - v))

        grad = tape.gradient(critic_loss, self.critic.get_variable())
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.get_variable()))

    # @tf.function
    def policy_train(self, IMG, action, advantage, old_prob):
        """

        :param state:
        :param action:
        :param advantage:
        :param old_prob: old actor net output
        :return:
        """
        with tf.GradientTape() as tape:
            # 计算新的网络分布
            mu, sigma = self.policy(IMG)
            pi = tfp.distributions.Normal(mu, sigma)

            # ratio = tf.clip_by_value(pi.prob(action), 0, 1) / (old_prob+1e-6)
            ratio = pi.prob(action) / (old_prob + 1e-6)
            del pi
            # ratio = old_prob

            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                )
            )

        actor_grad = tape.gradient(actor_loss, self.policy.get_variable())
        self.policy_optimizer.apply_gradients(zip(actor_grad, self.policy.get_variable()))
        return actor_loss

    @tf.function
    def get_loss(self, state, action, advantage, old_prob, discount_reward):

        mu, sigma = self.policy(state)
        pi = tfp.distributions.Normal(mu, sigma)

        ratio = pi.prob(action) / old_prob
        del pi

        v = self.critic.Model(state)
        surrogate1 = tf.square(v[1:] - discount_reward[1:])
        surrogate2 = tf.square(
            tf.clip_by_value(v[1:], v[:-1] - self.clip_ratio, v[:-1] + self.clip_ratio) - discount_reward[1:])
        critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        actor_loss = -tf.reduce_mean(
            tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
            )
        )

        return actor_loss, critic_loss

    @tf.function
    def get_v(self, IMG):
        return tf.squeeze(self.critic.Model(IMG))

    def optimize(self, batches):
        sum_rewards = []
        for batch in batches:
            sum_reward = np.sum(batch.reward_buffer)
            sum_rewards.append(sum_reward)
        sum_rewards = np.hstack(sum_rewards)
        info = {
            'max': np.max(sum_rewards),
            'min': np.min(sum_rewards),
            'avg': np.mean(sum_rewards)
        }
        print("Max episode reward:{}".format(info['max']))
        print("Min episode reward:{}".format(info["min"]))
        print("Average episode reward:{}".format(info["avg"]))

        IMG_buffer = np.concatenate([batch.IMG_buffer for batch in batches])
        next_IMG_buffer = np.concatenate([batch.next_IMG_buffer for batch in batches])
        reward_buffer = np.concatenate([batch.reward_buffer for batch in batches])
        action_buffer = np.concatenate([batch.action_buffer for batch in batches])
        old_probs = np.concatenate([batch.prob_buffer for batch in batches])

        tf_IMG_buffer = tf.cast(IMG_buffer, dtype=tf.float32)
        tf_next_IMG_buffer = tf.cast(next_IMG_buffer, dtype=tf.float32)

        values = self.get_v(tf_IMG_buffer).numpy()
        values_ = self.get_v(tf_next_IMG_buffer).numpy()

        sum_batch_rewards = []

        for i in range(self.span):
            path = slice(i * self.batch_size, (i + 1) * self.batch_size)
            sum_rewards = np.sum(reward_buffer[path])
            sum_batch_rewards.append(sum_rewards)

        print("Max mini_batch reward:{}".format(np.max(sum_batch_rewards)))
        print("Min mini_batch reward:{}".format(np.min(sum_batch_rewards)))
        print("Average mini_batch reward:{}".format(np.mean(sum_batch_rewards)))

        if self.reward_scale:
            reward_buffer = self.RewardScale(reward_buffer)

        gaes = []
        targets = []
        for i in range(self.span):
            path = slice(i * self.batch_size, (i + 1) * self.batch_size)
            gae, target = gae_target(self.gamma, self.lambada, reward_buffer[path], values[path],
                                     values_[i * self.batch_size - 1], done=False)

            gaes.append(gae)
            targets.append(target)
        gaes = np.concatenate([gae for gae in gaes])
        targets = np.concatenate([target for target in targets])

        if self.center_adv:
            gaes = standardize(gaes)

        action_buffer = tf.cast(action_buffer, dtype=tf.float32)
        gaes = tf.cast(gaes, dtype=tf.float32)
        old_probs = tf.cast(old_probs, dtype=tf.float32)
        targets = tf.cast(targets, dtype=tf.float32)

        for _ in tf.range(0, self.update_steps):
            for i in tf.range(0, self.mini_batch_size_num):
                path = slice(i * self.mini_batch_size, (i + 1) * self.mini_batch_size)
                IMG = IMG_buffer[path]
                action = action_buffer[path]
                gae = gaes[path]
                old_prob = old_probs[path]
                target = targets[path]
                self.policy_train(IMG, action, gae, old_prob)
                self.critic_train(IMG, target)
        del batches[:]
        del action_buffer
        del gaes
        del old_probs
        del targets
        return info

    def experiment_param(self, path):
        env_args = {
            'trajs': self.env_args.trajs,
            'steps': self.env_args.steps,
            'batch_size': self.env_args.batch_size,
            'mini_batch_size_num': self.env_args.mini_batch_size_num,
            'epochs': self.env_args.epochs
        }

        hyper_parameter = {
            'clip_ratio': self.hyper_parameter.clip_ratio,
            'policy_learning_rate': self.hyper_parameter.policy_learning_rate,
            'critic_learning_rate': self.hyper_parameter.critic_learning_rate,
            'update_steps': self.hyper_parameter.update_steps,
            'gamma': self.hyper_parameter.gamma,
            'lambada': self.hyper_parameter.lambada,
            'clip_value': self.hyper_parameter.clip_value,
            'reward_scale': self.hyper_parameter.reward_scale,
            'center': self.hyper_parameter.center,
            'scale': self.hyper_parameter.scale,
            'center_adv': self.hyper_parameter.center_adv
        }
        with open(path + '/env_args.yaml', 'w') as f:
            yaml.dump(env_args, f)
        with open(path + '/hyper_parameter.yaml', 'w') as f:
            yaml.dump(hyper_parameter, f)

    def optimize_MPI(self, observations, next_observations, actions, rewards, old_probs, reward_infos):
        sum_rewards = []
        sum_reward_infos = []
        for i in range(self.trajs * self.multi_worker_num):
            path = slice(i * self.steps, (i + 1) * self.steps)
            sum_reward = np.sum(rewards[path])
            sum_reward_info = np.sum(reward_infos[path], axis=0)
            # print(reward_infos[path])
            sum_rewards.append(sum_reward)
            sum_reward_infos.append(sum_reward_info)
        sum_rewards = np.hstack(sum_rewards)
        sum_reward_infos = np.vstack(sum_reward_infos)
        avg_info = np.mean(sum_reward_infos, axis=0)
        # print(sum_reward_infos)

        info = {
            'max': np.max(sum_rewards),
            'min': np.min(sum_rewards),
            'avg': np.mean(sum_rewards),
            'dot_p_x': avg_info[0],
            'dot_p_y': avg_info[1],
            'vel': avg_info[2]
        }
        print("Max episode reward:{}".format(info['max']))
        print("Min episode reward:{}".format(info["min"]))
        print("Average episode reward:{}".format(info["avg"]))
        del sum_rewards

        values = self.get_v(observations).numpy()
        values_ = self.get_v(next_observations).numpy()

        gaes = []
        targets = []
        for i in range(self.span):
            path = slice(i * self.batch_size, (i + 1) * self.batch_size)
            gae, target = gae_target(self.gamma, self.lambada, rewards[path], values[path],
                                     values_[i * self.batch_size - 1], done=False)

            gaes.append(gae)
            targets.append(target)
        gaes = np.concatenate([gae for gae in gaes])
        targets = np.concatenate([target for target in targets])

        if self.center_adv:
            gaes = standardize(gaes)

        action_buffer = tf.cast(actions, dtype=tf.float32)
        gaes = tf.cast(gaes, dtype=tf.float32)
        old_probs = tf.cast(old_probs, dtype=tf.float32)
        targets = tf.cast(targets, dtype=tf.float32)
        tf_observation_buffer = tf.cast(observations, dtype=tf.float32)

        for _ in tf.range(0, self.update_steps):
            for i in tf.range(0, self.mini_batch_size_num):
                path = slice(i * self.mini_batch_size, (i + 1) * self.mini_batch_size)
                state = tf_observation_buffer[path]
                action = action_buffer[path]
                gae = gaes[path]
                old_prob = old_probs[path]
                target = targets[path]
                self.policy_train(state, action, gae, old_prob)
                self.critic_train(state, target)

        del gaes
        del targets
        return info

    def train(self, path=None, title=None):
        if path is None:
            path = 'data'
            mkdir(path)
        else:
            mkdir(path)

        average_reward = []

        for i in range(self.epochs):
            print("---------------------obtain samples:{}---------------------".format(i))
            self.worker.update(self.policy, self.critic)

            time_start = time.time()
            batches = self.worker.runner()
            time_end = time.time()
            print('consuming time:{}'.format(time_end - time_start))

            info = self.optimize(batches)
            average_reward.append(info['avg'])
            self.policy.save_model(path)
            self.critic.save_model(path)
            print("----------------------------------------------------------")

        plt.plot(average_reward)
        plt.xlabel('episode')
        plt.ylabel('avg_reward')
        if title:
            plt.title(title)
        plt.show()

    def save_model(self, path):
        self.policy.save_model(path)
        self.critic.save_model(path)

    def save_weights(self, path):
        self.policy.save_weights(path)
        self.critic.save_weights(path)

    def update_param(self, clip_value, steps):
        self.clip_value = clip_value
        self.steps = steps
        self.batch_size = steps
        self.total_steps = self.trajs*self.multi_worker_num*self.steps
        self.span = int(self.total_steps/self.batch_size)
        print(self.span)
