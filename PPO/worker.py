from PPO.collector import Collector, MPI_Collector
import numpy as np


class Worker:

    def __init__(self, env, env_args, hyper_parameter):
        self.env = env
        self.trajs = env_args.trajs
        self.batch_size = env_args.batch_size
        self.steps = env_args.steps

        self.policy = None
        self.critic = None

        self.width = env_args.width
        self.action_dims = env_args.action_dims
        self.height = env_args.height
        self.channel = env_args.channel

        self.gamma = hyper_parameter.gamma
        self.lambada = hyper_parameter.lambada

    def update(self, policy, critic):
        """
        使用新的policy和critic
        :param policy:
        :param critic:
        :return:
        """

        self.policy = policy
        self.critic = critic

    def runner(self):
        # print('start')
        batches = []
        for i in range(self.trajs):
            # print('1')
            # collector = Collector(self.width, self.height, self.channel, self.steps)
            IMG, _ = self.env.reset()
            for t in range(self.steps):
                # print('3')
                action, prob = self.policy.get_action(IMG)
                # print('2')
                IMG_, reward, done, _ = self.env.step(action)
                # collector.store(IMG, IMG_, action, reward, prob)
                IMG = IMG_
            # batches.append(collector)

        return batches


class MPIWorker:

    def __init__(self, env, env_args, hyper_parameter, reward_info_size):
        self.env = env
        self.trajs = env_args.trajs
        self.batch_size = env_args.batch_size
        self.steps = env_args.steps
        self.trajs_steps = env_args.trajs_steps
        self.reward_info_size = reward_info_size

        self.policy = None
        self.critic = None

        self.width = env_args.width
        self.action_dims = env_args.action_dims
        self.height = env_args.height
        self.channel = env_args.channel

        self.gamma = hyper_parameter.gamma
        self.lambada = hyper_parameter.lambada

    def update(self, policy, critic):
        """
        使用新的policy和critic
        :param policy:
        :param critic:
        :return:
        """

        self.policy = policy
        self.critic = critic

    def runner(self):
        collector = MPI_Collector(self.width, self.height, self.channel, self.trajs_steps, self.reward_info_size)
        for i in range(self.trajs):
            IMG, _ = self.env.reset()
            for t in range(self.steps):
                action, prob = self.policy.get_action(IMG)
                IMG_, reward, done, reward_info = self.env.step(action)
                collector.store(IMG, IMG_, action, reward, prob, reward_info)
                IMG = IMG_

        return collector.IMG_buffer, collector.next_IMG_buffer, collector.action_buffer, collector.reward_buffer, collector.prob_buffer, collector.reward_info

    def update_param(self, steps):
        self.steps = steps
        self.trajs_steps = self.steps*self.trajs
