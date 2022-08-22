from mpi4py import MPI
import sys

sys.path.append('../')
import tensorflow as tf
import pybullet
import numpy as np

from CRandom import Drone_env
from BulletDrone.common.DataType import DroneParam, SimFlagDrone
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.Drone import Drone
from BulletDrone.control.SimplePID import PIDParam
from BulletDrone.utils.FileTools import rename_itr, copy_file

import time
from PPO.worker import MPIWorker
from RL.common import args
from RL.common.functions import mkdir
from PPO.Eargs import RewardParameter
from utils import RewardParameter, TrainingProcessController
from PPO.Model import Model1, V_Model
from PPO.GaussianPolicy3 import gaussian_policy
from PPO.CriticPolicy import critic_policy
from PPO.PPO import PPO
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    tf.random.set_seed(1)
# else:
#     # seed = rank*0.01
#     tf.random.set_seed(rank)
#     np.random.seed(rank)

if rank == 0:
    import os

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")

else:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ----------------training parameter-----------------
reward_info_size = 3
cache_path = 'test'  # model's name
load_path = 'trier5'

env_args1 = args.IMGEnvArgs(trajs=8, steps=150, epochs=200000, batch_size=150, mini_batch_size_num=4, width=64,
                            height=64,
                            channel=1, action_dims=3, multi_worker_num=size - 1)

hyper_parameters = args.HyperParameter(clip_ratio=0.2, policy_learning_rate=3e-5, critic_learning_rate=1e-5,
                                       update_steps=4, gamma=0.95, lambada=1, scale=False, center=False,
                                       reward_scale=False, clip_value=True, center_adv=False, )

reward_param = RewardParameter(pos_xy=100, z=1, vel=0, acc=0, omega=0, rot=0, special_reward=False, done_r=10,
                               not_done=-5, stop_enable=False)

F450_param = DroneParam(prop_pos=np.array([0.1597, 0.1597, 0]), inertia=np.array([1.745e-2, 1.745e-2, 3.175e-2]),
                        Ct=1.105e-5, Cm=1.489e-7, Cd=6.579e-2, Tm=0.0136, mass=1.5, max_rpm=7408.7, max_speed=13.4,
                        max_tilt_angle=0.805, freq_GPG=10)

# training control
stages = 8
reward_threshold_times = 10
steps_dic = [200, 200, 200, 200, 200, 200, 200, 200]
reward_threshold = [170, 170, 170, 170, 170, 170, 170, 200]
random_pictures_dic = [2, 2, 3, 4, 5, 6, 7, 8]
random_land_color_dic = [False, False, False, False, False, False, False, False]
random_land_size_dic = [False, False, False, True, True, True, True, True]
random_land_or_dic = [0.5, 0.4, 0.5, 0.6, 0.7, 1.0, 1.0, 1.0]
noise_land_dic = [True, True, True, True, True, True, True, True]
clip_value_dic = [0.02, 0.06, 0.05, 0.05, 0.05, 0.05, 0.03, 0.01]
init_random_pos_dic = [2, 2, 2, 2, 2, 2, 2, 2]
reward_pos_xy_dic = [100, 100, 100, 100, 100, 100, 100, 100]
reward_pos_z_dic = [2, 3, 3, 3, 3, 2, 3, 3]
reward_special_flag_dic = [False, False, False, False, False, False, False, False]
train_progress_control = TrainingProcessController(stages, reward_threshold, reward_threshold_times, steps_dic,
                                                   random_pictures_dic, random_land_color_dic, random_land_size_dic,
                                                   random_land_or_dic, noise_land_dic, clip_value_dic,
                                                   init_random_pos_dic, reward_pos_xy_dic, reward_pos_z_dic,
                                                   reward_special_flag_dic)
# ----------------------------------------------------

F450_pid_param = None
F450_sim_flag = None
bullet_manager = None
F450_drone = None
env = None
worker = None

if rank != 0:
    F450_pid_param = PIDParam(
        Kph=0,
        Kpz=0,
        KVhd=0,
        KVhp=2,
        KVhi=0,
        KVzp=3,
        KVzi=0,
        KVzd=0,
        KOmega_x=10,
        KOmega_y=10,
        KOmega_z=15,
        Kwp_x=2,
        Kwp_y=2,
        Kwp_z=3,
        Kwi_x=0.8,
        Kwi_y=0.8,
        Kwi_z=2.5,
        Kwd_x=0,
        Kwd_y=0,
        Kwd_z=0,
        acc_hmax=0,
        acc_zmax=0,
        delta_hmax=0,
        delta_zmax=0,
        m=F450_param.mass,
        inertia=F450_param.inertia
    )
    F450_sim_flag = SimFlagDrone(
        motor_lag=False,
        sensor_noise=True,
        pwm_noise=False,
        GPS_lag=False
    )
    bullet_manager = BulletManager(
        connection_mode='GUI',
    )
    # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING)
    # bullet_manager.load_background('plane100.urdf')
    # bullet_manager.load_background('src/cube2.urdf')
    # F45_id = bullet_manager.load_drone('src/F450.urdf')
    F450_drone = Drone(2, F450_param, F450_sim_flag)
    env = Drone_env(
        drone=F450_drone,
        pid_param=F450_pid_param,
        bullet_manager=bullet_manager,
        reward_param=reward_param,
        interval=int(133 / size),
        index=rank
    )
    worker = MPIWorker(
        env=env,
        env_args=env_args1,
        hyper_parameter=hyper_parameters,
        reward_info_size=reward_info_size
    )

actor = Model1(64, 64)

value_net = V_Model(64, 64)

IMG = np.random.normal(size=(1, 64, 64, 1))
actor(IMG)
value_net(IMG)
policy = gaussian_policy(Model=actor, file_name='policy')
critic = critic_policy(Model=value_net, file_name='critic')

if rank != 0:
    worker.update(policy, critic)

log_dir = None
file_writer = None
ppo = None

start = None
end = None
history_model_path = None
total_time = 0
save_freq = 3

if rank == 0:
    ppo = PPO(
        policy=policy,
        critic=critic,
        env_args=env_args1,
        hyper_parameter=hyper_parameters,
        net_visualize=False
    )
    mkdir(cache_path)
    log_dir = cache_path + '/logs'
    file_writer = tf.summary.create_file_writer(log_dir)
    history_model_path = cache_path + '/model'
    mkdir(history_model_path)
    for i in range(stages):
        mkdir(history_model_path + '/' + 'stage{}'.format(i))
    # if os.path.exists("workspace"):
    #     os.rmdir("workspace")
    mkdir('workspace')
    # os.makedirs("workspace/scene")
    for i in range(size - 1):
        mkdir('workspace/' + str(i + 1))
        mkdir('workspace/' + str(i + 1) + '/noise')
        copy_file('source', 'workspace/' + str(i + 1), 'plane.mtl')
        copy_file('source', 'workspace/' + str(i + 1), 'plane.obj', 'plane0.obj')
        copy_file('source', 'workspace/' + str(i + 1), 'plane.urdf', 'plane0.urdf')
    ppo.experiment_param(cache_path)
    if load_path:
        policy.load_weights(load_path)
        critic.load_weights(load_path)
comm.Barrier()

if rank == 1:
    worker.env.reward_parameter(cache_path)

for i in range(env_args1.epochs):
    recv_observations = None
    recv_next_observations = None
    recv_actions = None
    recv_rewards = None
    recv_old_probs = None
    recv_reward_infos = None
    std = None
    avg_reward = 0

    if rank > 0:
        std = np.empty(env_args1.action_dims)
        avg_reward = np.empty(1)

    # training control

    if rank == 0:
        clip_value, steps = train_progress_control.ppo_get()
        ppo.update_param(clip_value, steps)
        if train_progress_control.init_log_std_flag:
            train_progress_control.reset()
            ppo.policy.init_log_std()

    else:
        worker.update_param(train_progress_control.worker_get())

        random_pictures, land_color_flag, land_size_flag, land_or, noise_land_flag, init_random_pos = train_progress_control.env_get()
        pos_xy, z, special_reward = train_progress_control.reward_get()
        env.update_param(random_pictures, land_color_flag, land_size_flag, land_or, noise_land_flag, init_random_pos,
                         pos_xy, z, special_reward)
        if train_progress_control.init_log_std_flag:
            train_progress_control.reset()

    if rank == 0:
        start = time.time()
        print('---------------------stage:{}--obtain samples:{}---------------------'.format(
            train_progress_control.current_stage, i))
        ppo.save_weights(cache_path)

        std = ppo.policy.get_std()
        print('current std:{}'.format(std))
        recv_observations = np.empty(
            [size, train_progress_control.current_steps * env_args1.trajs, env_args1.width, env_args1.height,
             env_args1.channel])
        recv_next_observations = np.empty(
            [size, train_progress_control.current_steps * env_args1.trajs, env_args1.width, env_args1.height,
             env_args1.channel])
        recv_actions = np.empty([size, train_progress_control.current_steps * env_args1.trajs, env_args1.action_dims])
        recv_rewards = np.empty([size, train_progress_control.current_steps * env_args1.trajs, 1])
        recv_old_probs = np.empty([size, train_progress_control.current_steps * env_args1.trajs, env_args1.action_dims])
        recv_reward_infos = np.empty([size, train_progress_control.current_steps * env_args1.trajs, reward_info_size])

    comm.Barrier()
    comm.Bcast(std)

    if rank == 0:
        send_observations = np.zeros((train_progress_control.current_steps * env_args1.trajs, env_args1.width,
                                      env_args1.height, env_args1.channel))
        send_next_observations = np.zeros((train_progress_control.current_steps * env_args1.trajs, env_args1.width,
                                           env_args1.height, env_args1.channel))
        send_actions = np.zeros((train_progress_control.current_steps * env_args1.trajs, env_args1.action_dims))
        send_rewards = np.zeros((train_progress_control.current_steps * env_args1.trajs, 1))
        send_old_probs = np.zeros((train_progress_control.current_steps * env_args1.trajs, env_args1.action_dims))
        send_reward_infos = np.zeros((train_progress_control.current_steps * env_args1.trajs, reward_info_size))
    else:
        policy.load_weights(cache_path)
        critic.load_weights(cache_path)

        policy.set_std(std)

        worker.update(policy, critic)
        send_observations, send_next_observations, send_actions, send_rewards, send_old_probs, send_reward_infos = worker.runner()

    comm.Gather(send_observations, recv_observations, root=0)
    comm.Gather(send_next_observations, recv_next_observations, root=0)
    comm.Gather(send_actions, recv_actions, root=0)
    comm.Gather(send_rewards, recv_rewards, root=0)
    comm.Gather(send_old_probs, recv_old_probs, root=0)
    comm.Gather(send_reward_infos, recv_reward_infos, root=0)
    # print(recv_actions)
    comm.Barrier()

    if rank == 0:
        observations = np.concatenate(recv_observations[1:], axis=0)
        next_observations = np.concatenate(recv_next_observations[1:], axis=0)
        actions = np.concatenate(recv_actions[1:], axis=0)
        rewards = np.concatenate(recv_rewards[1:], axis=0)
        old_probs = np.concatenate(recv_old_probs[1:], axis=0)
        reward_infos = np.concatenate(recv_reward_infos[1:], axis=0)
        info = ppo.optimize_MPI(observations, next_observations, actions, rewards, old_probs, reward_infos)
        avg_reward = np.array([info['avg']])
        # avg_reward_info = np.sum(reward_infos, axis=0)
        print("sum_dposx:{}, sum_vel:{}".format(info['dot_p_x'], info['vel']))
        print("sum_dposy:{}".format(info['dot_p_y']))
        with file_writer.as_default():
            tf.summary.scalar("Reward/Average reward", info['avg'], step=i)
            tf.summary.scalar("Reward/Max reward", info['max'], step=i)
            tf.summary.scalar("Reward/Min reward", info["min"], step=i)
            tf.summary.scalar("Reward/dot_pos_x", info['dot_p_x'], step=i)
            tf.summary.scalar("Reward/dot_pos_y", info['dot_p_y'], step=i)
            tf.summary.scalar("Reward/dot_vel", info['vel'], step=i)

        ppo.save_weights(cache_path)
        if i % save_freq == 0:
            model_save_path = history_model_path + '/' + 'stage{}/'.format(train_progress_control.current_stage) + str(
                i)
            mkdir(model_save_path)
            ppo.policy.save_weights(model_save_path)
            ppo.critic.save_weights(model_save_path)
        # print(observations.shape)
        end = time.time()
        t = end - start
        total_time = t + total_time
        print('consuming time:{}'.format(t))
        print('total tome:{}'.format(total_time))
        print("----------------------------------------------------------")

    comm.Bcast(avg_reward)
    train_progress_control.control_policy(avg_reward, i)

    comm.barrier()
