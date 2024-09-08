import torch
from collections import deque
import numpy as np
import logging
from dataclasses import dataclass

from torch.distributions import Categorical
from tqdm import trange
import logging

from Code.Module.EEG_Encoder.TF_Encoder import TF_Encoder_Critic, TF_Encoder_Actor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.autograd.set_detect_anomaly(True)
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal

@dataclass
class PPOCfg:
    gamma: float = 0.99  # 折扣因子
    k_epochs: int = 4  # 更新策略网络的次数
    actor_lr: float = 0.0003  # actor网络的学习率
    critic_lr: float = 0.0003  # critic网络的学习率
    eps_clip: float = 0.2  # epsilon-clip
    entropy_coef: float = 0.01  # entropy的系数
    update_freq: int = 100  # 更新频率
    device: str = "cuda"  # 使用的设备


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

        self.device = torch.device(args.device)

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def clear(self, idx):
        for i in range(idx):
            self.count -= 1
            self.s[self.count] = np.zeros(self.args.state_dim)
            self.a[self.count] = np.zeros(self.args.action_dim)
            self.a_logprob[self.count] = np.zeros(self.args.action_dim)
            self.r[self.count] = np.zeros(1)
            self.s_[self.count] = np.zeros(self.args.state_dim)
            self.dw[self.count] = np.zeros(1)
            self.done[self.count] = np.zeros(1)

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float, device=self.device)
        a = torch.tensor(self.a, dtype=torch.float, device=self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float, device=self.device)
        r = torch.tensor(self.r, dtype=torch.float, device=self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float, device=self.device)
        dw = torch.tensor(self.dw, dtype=torch.float, device=self.device)
        done = torch.tensor(self.done, dtype=torch.float, device=self.device)

        return s, a, a_logprob, r, s_, dw, done


class PPO_continuous:
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        # self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.device = torch.device(args.device)

        if self.policy_dist == "Beta":
            # self.actor = Actor_Beta(args)
            # TODO: Implement Actor_Beta
            pass
        else:
            self.actor = TF_Encoder_Actor(args.action_dim).to(self.device)

        self.critic = TF_Encoder_Critic(1).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        with torch.no_grad():
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
            if self.policy_dist == "Beta":
                a = self.actor.mean(s).detach().numpy().flatten()
            else:
                # a = self.actor(s)[0].cpu().detach().numpy().flatten()
                eval_dist = self.actor.getDist(s)
                a = eval_dist.sample().cpu().detach().numpy().flatten()
            return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        # print(s)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.getDist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                # a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float, device=self.device).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in trange(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.getDist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
    
    def save(self, path):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }
        torch.save(save_dict, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print("Successfully load the model from", path)

if __name__ == '__main__':
    cfg = PPOCfg()
    agent = Agent(cfg)

    s = torch.ones(1, 1000)

    print(s.shape)
    action, log_probs = agent.sample_action(s)
