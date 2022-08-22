from mpi4py import MPI
import sys

sys.path.append('../')
import tensorflow as tf
import pybullet
import numpy as np

from Camera2 import Drone_env
from BulletDrone.common.DataType import DroneParam, SimFlagDrone
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.Drone import Drone
from BulletDrone.control.SimplePID import PIDParam

import time
from PPO.worker import MPIWorker
from RL.common import args
from RL.common.functions import mkdir
from PPO.Eargs import RewardParameter
from args import RewardParameter
from PPO.Model import Model1, V_Model
from PPO.GaussianPolicy3 import gaussian_policy
from PPO.CriticPolicy import critic_policy
from PPO.PPO import PPO

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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

reward_info_size = 3
cache_path = 'end2end8'

F450_param = DroneParam(
    prop_pos=np.array([0.1597, 0.1597, 0]),
    inertia=np.array([1.745e-2, 1.745e-2, 3.175e-2]),
    Ct=1.105e-5,
    Cm=1.489e-7,
    Cd=6.579e-2,
    Tm=0.0136,
    mass=1.5,
    max_rpm=7408.7,
    max_speed=13.4,
    max_tilt_angle=0.805,
    freq_GPG=10
)

reward_param = RewardParameter(pos_xy=100, z=1, vel=0, acc=0, omega=0, rot=0, special_reward=True, done_r=10,
                               not_done=-5, stop_enable=False)

env_args1 = args.IMGEnvArgs(
    trajs=3,
    steps=400,
    epochs=2000,
    batch_size=300,
    mini_batch_size_num=4,
    width=64,
    height=64,
    channel=1,
    action_dims=3,
    multi_worker_num=size - 1
)

hyper_parameters = args.HyperParameter(
    clip_ratio=0.2,
    policy_learning_rate=3e-5,
    critic_learning_rate=1e-5,
    update_steps=4,
    gamma=0.95,
    lambada=1,
    scale=False,
    center=False,
    reward_scale=False,
    clip_value=True,
    center_adv=False,
)

if rank == 0:
    tf.random.set_seed(1)
# else:
#     # seed = rank*0.01
#     tf.random.set_seed(rank)
#     np.random.seed(rank)



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
    bullet_manager.load_background('plane100.urdf')
    bullet_manager.load_background('src/cube2.urdf')
    F45_id = bullet_manager.load_drone('src/F450.urdf')
    F450_drone = Drone(F45_id, F450_param, F450_sim_flag)
    env = Drone_env(
        drone=F450_drone,
        pid_param=F450_pid_param,
        bullet_manager=bullet_manager,
        reward_param=reward_param
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
    ppo.experiment_param(cache_path)
    policy.load_weights('end2end8_4/model/321')
    critic.load_weights('end2end8_4/model/321')
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

    if rank > 0:
        std = np.empty(env_args1.action_dims)

    if rank == 0:
        start = time.time()
        print("---------------------obtain samples:{}---------------------".format(i))
        ppo.save_weights(cache_path)

        std = policy.get_std()
        print('current std:{}'.format(std))
        recv_observations = np.empty(
            [size, env_args1.trajs_steps, env_args1.width, env_args1.height, env_args1.channel])
        recv_next_observations = np.empty(
            [size, env_args1.trajs_steps, env_args1.width, env_args1.height, env_args1.channel])
        recv_actions = np.empty([size, env_args1.trajs_steps, env_args1.action_dims])
        recv_rewards = np.empty([size, env_args1.trajs_steps, 1])
        recv_old_probs = np.empty([size, env_args1.trajs_steps, env_args1.action_dims])
        recv_reward_infos = np.empty([size, env_args1.trajs_steps, reward_info_size])

    comm.Barrier()
    comm.Bcast(std)

    if rank == 0:
        send_observations = np.zeros((env_args1.trajs_steps, env_args1.width, env_args1.height, env_args1.channel))
        send_next_observations = np.zeros((env_args1.trajs_steps, env_args1.width, env_args1.height, env_args1.channel))
        send_actions = np.zeros((env_args1.trajs_steps, env_args1.action_dims))
        send_rewards = np.zeros((env_args1.trajs_steps, 1))
        send_old_probs = np.zeros((env_args1.trajs_steps, env_args1.action_dims))
        send_reward_infos = np.zeros((env_args1.trajs_steps, reward_info_size))
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
            model_save_path = history_model_path + '/' + str(i)
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
    comm.barrier()
