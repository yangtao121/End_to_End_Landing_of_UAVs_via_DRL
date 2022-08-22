import numpy as np


class RewardParameter:
    def __init__(self, pos_xy, z, vel, acc, omega, rot, special_reward=False, done_r=5, not_done=-2, stop_enable=True):
        self.pos_xy = pos_xy
        self.z = z
        self.vel = vel
        self.acc = acc
        self.omega = omega
        self.rot = rot
        self.special_reward = special_reward
        self.done_r = done_r
        self.not_done = not_done
        self.stop_enable = stop_enable

    def update_param(self, pos_xy, z, special_reward):
        self.pos_xy = pos_xy
        self.z = z
        self.special_reward = special_reward


class TrainingProcessController:
    """
    Control training phase.
    You can redesign this class based on your need.
    """

    def __init__(self, stages, reward_threshold_dic, reward_hold_times, steps_dic, random_pictures_dic,
                 random_land_color_dic, random_land_size_dic, random_land_or_dic, noise_land_dic, clip_value_dic,
                 init_random_pos_dic, reward_pos_xy_dic, reward_pos_z_dic, reward_special_flag_dic):
        """
        :param
        reward_threshold: the best reward you want.
        reward_hold_times: keep the reward_threshold times.
        trajs: init trajs.
        trajs_dic: the dim is stages
        random_pictures_dic: the dim is stages
        random_land_color_dic: the dim is stages. is flag
        """
        self.stages = stages
        self.reward_hold_times = reward_hold_times

        self.reward_container = -100 * np.ones(self.reward_hold_times)
        self.current_stage = 0

        self.current_reward_threshold = reward_threshold_dic[self.current_stage]
        self.reward_threshold_dic = reward_threshold_dic

        self.steps_dic = steps_dic
        self.current_steps = steps_dic[self.current_stage]

        self.random_pictures = random_pictures_dic
        self.current_random_pictures = random_pictures_dic[self.current_stage]

        self.random_land_color_dic = random_land_color_dic
        self.current_land_color_flag = random_land_color_dic[self.current_stage]

        self.random_land_size_dic = random_land_size_dic
        self.current_land_size_flag = random_land_size_dic[self.current_stage]

        self.random_land_or_dic = random_land_or_dic
        self.current_land_or = random_land_or_dic[self.current_stage]

        self.noise_land_dic = noise_land_dic
        self.current_noise_land_flag = noise_land_dic[self.current_stage]

        self.current_clip_value = clip_value_dic[self.current_stage]
        self.clip_value_dic = clip_value_dic

        self.current_init_random_pos = init_random_pos_dic[self.current_stage]
        self.init_random_pos_dic = init_random_pos_dic

        self.init_log_std_flag = False

        self.current_reward_pos_xy = reward_pos_xy_dic[self.current_stage]
        self.reward_pos_xy_dic = reward_pos_xy_dic

        self.current_reward_pos_z = reward_pos_z_dic[self.current_stage]
        self.reward_pos_z_dic = reward_pos_z_dic

        self.current_reward_special_flag = reward_special_flag_dic[self.current_stage]
        self.reward_special_flag_dic = reward_special_flag_dic

    def control_policy(self, reward, epoch):
        """
        This function define the policy
        """
        self.reward_container[epoch % self.reward_hold_times] = reward
        self.init_log_std_flag = False
        if np.mean(self.reward_container) > self.current_reward_threshold:
            if self.current_stage != self.stages:
                self.current_stage += 1
            self.current_steps = self.steps_dic[self.current_stage]
            self.current_random_pictures = self.random_pictures[self.current_stage]

            self.current_land_color_flag = self.random_land_color_dic[self.current_stage]
            self.current_land_size_flag = self.random_land_size_dic[self.current_stage]
            self.current_land_or = self.random_land_or_dic[self.current_stage]
            self.current_noise_land_flag = self.noise_land_dic[self.current_stage]
            self.current_init_random_pos = self.init_random_pos_dic[self.current_stage]

            self.current_reward_pos_xy = self.reward_pos_xy_dic[self.current_stage]
            self.current_reward_pos_z = self.reward_pos_z_dic[self.current_stage]
            self.current_reward_special_flag = self.reward_special_flag_dic[self.current_stage]
            self.current_clip_value = self.clip_value_dic[self.current_stage]

            self.init_log_std_flag = True

            self.current_reward_threshold = self.reward_threshold_dic[self.current_stage]

    def worker_get(self):
        return self.current_steps

    def env_get(self):
        """
        return:
        random_pictures, land_color_flag,land_size_flag,land_or
        """
        return self.current_random_pictures, self.current_land_color_flag, self.current_land_size_flag, self.current_land_or, self.current_noise_land_flag, self.current_init_random_pos

    def ppo_get(self):
        return self.current_clip_value, self.current_steps

    def reward_get(self):
        return self.current_reward_pos_xy, self.current_reward_pos_z, self.current_reward_special_flag

    def reset(self):
        self.reward_container = -100 * np.ones(self.reward_hold_times)

