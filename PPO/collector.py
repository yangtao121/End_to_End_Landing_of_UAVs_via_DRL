import numpy as np


class Collector:
    """
    维度上可能有问题，需要重新选择。
    """

    def __init__(self,
                 width,
                 height,
                 channel,
                 episode_length
                 ):
        self.IMG_buffer = np.zeros([episode_length, width, height, channel])
        self.state_buffer = np.zeros([episode_length, 5, 18])
        self.next_IMG_buffer = np.zeros([episode_length, width, height, channel])
        self.next_state_buffer = np.zeros([episode_length, 5, 18])

        self.action_buffer = np.zeros([episode_length, 3])
        self.reward_buffer = np.zeros([episode_length, 1])
        self.prob_buffer = np.zeros([episode_length, 3])
        self.pointer = 0
        self.last_pointer = 0

    def store(self, img, next_img, state, next_state, action, reward, prob):
        self.IMG_buffer[self.pointer] = img
        self.next_IMG_buffer[self.pointer] = next_img
        self.state_buffer[self.pointer] = state
        self.next_state_buffer[self.pointer] = next_state

        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.prob_buffer[self.pointer] = prob

        self.pointer += 1


class MPI_Collector:
    def __init__(self, width, height, channel, length, reward_size):
        self.IMG_buffer = np.zeros([length, width, height, channel])
        self.next_IMG_buffer = np.zeros([length, width, height, channel])

        self.action_buffer = np.zeros([length, 3])
        self.reward_buffer = np.zeros([length, 1])
        self.prob_buffer = np.zeros([length, 3])

        self.reward_info = np.zeros([length, reward_size])

        self.pointer = 0

    def store(self, img, next_img, action, reward, prob, reward_info):
        self.IMG_buffer[self.pointer] = img
        self.next_IMG_buffer[self.pointer] = next_img

        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.prob_buffer[self.pointer] = prob
        self.reward_info[self.pointer] = reward_info

        self.pointer += 1
