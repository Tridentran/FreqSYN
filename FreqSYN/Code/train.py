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
from Code.Utils.Utils import load_json, add_args, load_cfg_args
# import time

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
    

def evaluate_policy(args, env, agent, state_norm):
    evaluate_reward = 0
    for t in trange(args.evaluate_times):
        s, done, _ = env.reset(reset_mode="test", gen_mode=1, build_gen=t)
        s = s.cpu().numpy().flatten()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward= 0
        times = 0
        while (not done) and (times < args.max_episode_steps):
            if times > 10:
                assert 0
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            s_ = s_.cpu().numpy().flatten()
            r = np.float64(r)
            episode_reward += r
            s = s_
            times += 1
        evaluate_reward += episode_reward

    return evaluate_reward / args.evaluate_times


def main(args, env, number, seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_name = "EEG_AUG_Env"
    args.state_dim = 1000
    args.action_dim = int(args.action_dim)
    args.max_episode_steps = 10  # Maximum number of steps per episode


    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))


    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    log_dir = os.path.join(os.getcwd(), "Log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'env_{}_{}_dataset_{}_name_{}'.format(env_name, args.policy_dist,args.dataset, number,)))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # param in our method
    gen_eval = 0    # after eval will be set to 0, which means that we should rebuild the env for train
    best_r = -10000000

    with tqdm(total=args.max_train_steps) as pbar:
        while total_steps < args.max_train_steps:
            s, done, _ = env.reset(reset_mode="train", gen_mode=0, build_gen=gen_eval)

            if gen_eval == 0:
                gen_eval = -1
            s = s.cpu().numpy().flatten()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            while not done and episode_steps < args.max_episode_steps:
                episode_steps += 1
                a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) 
                else:
                    action = a
                s_, r, done = env.step(action)
                s_ = s_.cpu().numpy().flatten()
                r = np.float64(r)

                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1
                pbar.update(1)

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % args.evaluate_freq == 0:
                    evaluate_num += 1
                    evaluate_reward = evaluate_policy(args, env, agent, state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    tqdm.write("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                    # Save the rewards
                    save_root_path = os.path.join(working_dir, f"/Model/{args.dataset}/PPO-{number}")
                    best_save_path = f"{evaluate_num}-{evaluate_rewards[-1]}-best.pth"
                    temp_save_path = f"{evaluate_num}-{evaluate_rewards[-1]}.pth"
                    if not os.path.exists(save_root_path):
                        os.makedirs(save_root_path)

                    if evaluate_rewards[-1] > best_r:
                        agent.save(os.path.join(save_root_path, best_save_path))
                        best_r = evaluate_rewards[-1]

                    if evaluate_num % args.save_freq == 0:
                        agent.save(os.path.join(save_root_path, temp_save_path))
                    
                    gen_eval = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    add_args(parser)
    args = parser.parse_args()

    # load json
    cfg = load_json(args.train_cfg)
    train_cfg = TrainConfig(**cfg['train_cfg'])
    load_cfg_args(args, train_cfg)

    # build env
    env_cfg = EEG_AUG_Env.EnvCfg(**cfg['env_cfg'])
    args.dataset = env_cfg.data_name
    env = EEG_AUG_Env.EEGAugEnv(env_cfg, args)
    main(args, env=env, number=args.train_name, seed=10)
