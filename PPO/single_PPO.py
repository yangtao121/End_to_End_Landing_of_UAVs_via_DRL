from PPO.PPO import PPO
from RL.common.functions import mkdir
import time
import tensorflow as tf


class PPO_bullet(PPO):
    def __init__(self,
                 policy,
                 critic,
                 worker,
                 env_args,
                 hyper_parameter,
                 net_visualize=False
                 ):
        super().__init__(policy, critic, env_args, hyper_parameter, worker, net_visualize)

    def train(self, drone_file, path=None, load_path=None, save_freq=None):
        # 创建文件夹
        if path is None:
            path = 'data'
            mkdir(path)
        else:
            mkdir(path)

        log_dir = path + '/logs'
        file_writer = tf.summary.create_file_writer(log_dir)
        file_writer.set_as_default()
        # cache_path = path + '/cache'  # 多线程模型加载区域
        history_model_path = path + '/model'
        # history_recorder_name = path + "/progress.csv"
        # mkdir(cache_path)
        mkdir(history_model_path)
        self.experiment_param(path)
        self.worker.env.reward_parameter(path)

        # 加载模型
        if load_path:
            self.policy.load_model(load_path + 'policy.h5')
            self.critic.load_model(load_path + 'critic.h5')

        # 创建csv表格用于保存每次的结果
        # header = ['Episode', 'Max Reward', 'Min Reward', 'Average Reward']
        # csv_file = open(history_recorder_name, 'w', newline='')
        # csv_writer = csv.writer(csv_file)
        # csv_writer.writerow(header)

        total_time = 0

        for i in range(self.epochs):
            print("---------------------obtain samples:{}---------------------".format(i))
            time_start = time.time()
            self.worker.update(self.policy, self.critic)

            batches = self.worker.runner()

            info = self.optimize(batches)
            tf.summary.scalar("Average reward", info['avg'], step=i)
            tf.summary.scalar("Max reward", info['max'], step=i)
            tf.summary.scalar("Min reward", info["min"], step=i)
            # csv_writer.writerow([i, info['max'], info['min'], info["avg"]])
            if save_freq:
                if i % save_freq == 0:
                    model_save_path = history_model_path + '/' + str(i)
                    mkdir(model_save_path)
                    self.policy.save_weights(model_save_path)
                    self.critic.save_weights(model_save_path)
            time_end = time.time()
            t = time_end - time_start
            total_time = t + total_time
            print('episode consuming time:{}'.format(t))
            print('consuming time:{}'.format(total_time))
            print("----------------------------------------------------------")
