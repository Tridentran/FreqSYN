import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse

from dataclasses import dataclass
import json
import sys
import os
import argparse
from tqdm import tqdm, trange

working_dir = os.getcwd()
sys.path.append(working_dir)

import Code.Env.EEG_AUG_Env_wo_batch as EEG_AUG_Env
from Code.RL.PPO.PPO_New import PPO_continuous, ReplayBuffer
from Code.RL.PPO.normalization import Normalization, RewardScaling
from Code.Utils.Utils import add_args, load_cfg_args, load_json


AUG_DATA_ROOT_PATH = os.path.join(working_dir, 'Data/AUG_DATA')
if not os.path.exists(AUG_DATA_ROOT_PATH):
    os.makedirs(AUG_DATA_ROOT_PATH)

@dataclass
class TrainConfig:
    # Maximum number of training steps
    max_train_steps: int = int(3e6)
    # Evaluate the policy every 'evaluate_freq' steps
    evaluate_freq: float = 5e3
    evaluate_times: int = 3
    # Save frequency
    save_freq: int = 20
    # Beta or Gaussian
    policy_dist: str = "Gaussian"
    # Batch size
    batch_size: int = 2048
    # Minibatch size
    mini_batch_size: int = 64
    # Learning rate of actor
    lr_a: float = 3e-4
    # Learning rate of critic
    lr_c: float = 3e-4
    # Discount factor
    gamma: float = 0.99
    # GAE parameter
    lamda: float = 0.95
    # PPO clip parameter
    epsilon: float = 0.2
    # PPO parameter
    K_epochs: int = 10
    device: str = "cuda"
    

# def main(args, env, number, seed):
def aug_data(args, env, agent_path, seed=10):
    np.random.seed()
    torch.manual_seed(seed)

    env_name = "EEG_AUG_Env"
    args.state_dim = 1000
    args.action_dim = int(args.action_dim)
    args.max_episode_steps = 10  # Maximum number of steps per episode


    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    agent = PPO_continuous(args)

    agent.load(agent_path)

    # state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    # if args.use_reward_norm:  # Trick 3:reward normalization
    #     reward_norm = Normalization(shape=1)
    # elif args.use_reward_scaling:  # Trick 4:reward scaling
    #     reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    agent.actor.eval()
    augmented_data = []
    data_label=[]
    action_list = []

    for i in trange(len(env.data['train'][0])):
        state, done, label = env.reset(reset_mode="train", gen_mode=1, build_gen=i)
        state = state.numpy().flatten()
        action = agent.evaluate(state)
        next_state, reward, done = env.step(action)
        augmented_data.append(next_state.cpu().numpy())
        data_label.append(label.cpu().numpy())
        action_list.append(action)

        if i == -1:
            break
    
    augmented_data = np.concatenate(augmented_data)
    augmented_labels = np.array(data_label)
    action_list = np.array(action_list)

    return augmented_data, augmented_labels, action_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    add_args(parser)
    parser.add_argument("--agent_path", type=str, help="agent file path")
    args = parser.parse_args()

    # load json
    cfg = load_json(args.train_cfg)
    train_cfg = TrainConfig(**cfg['train_cfg'])
    load_cfg_args(args, train_cfg)

    # build env
    env_cfg = EEG_AUG_Env.EnvCfg(**cfg['env_cfg'])
    args.dataset = env_cfg.data_name
    env = EEG_AUG_Env.EEGAugEnv(env_cfg, args)

    

    agent_path = args.agent_path
    DATA_SET_NAME = agent_path.split('/')[-3]
    AUG_DATA_NAME = agent_path.split('/')[-2].split('-')[1:]
    AUG_DATA_NAME = "-".join(AUG_DATA_NAME)

    augmented_eeg_data, augmented_eeg_labels, action = aug_data(args, env, agent_path, seed=10)

    if not os.path.exists(os.path.join(AUG_DATA_ROOT_PATH, DATA_SET_NAME, AUG_DATA_NAME)):
        os.makedirs(os.path.join(AUG_DATA_ROOT_PATH, DATA_SET_NAME, AUG_DATA_NAME))

    # 保存增强后的数据
    augmented_eeg_data_path = os.path.join(AUG_DATA_ROOT_PATH, DATA_SET_NAME, AUG_DATA_NAME, f'data.npy')
    augmented_eeg_labels_path = os.path.join(AUG_DATA_ROOT_PATH, DATA_SET_NAME, AUG_DATA_NAME, f'label.npy')
    action_path = os.path.join(AUG_DATA_ROOT_PATH, DATA_SET_NAME, AUG_DATA_NAME,f'action.npy')
    
    np.save(augmented_eeg_data_path, augmented_eeg_data)
    np.save(augmented_eeg_labels_path, augmented_eeg_labels)
    np.save(action_path, action)