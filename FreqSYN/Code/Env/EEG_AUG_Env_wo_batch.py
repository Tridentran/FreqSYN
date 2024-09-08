import copy
from itertools import cycle
import math
from typing import Tuple

import numpy as np
from dataclasses import dataclass
from scipy import signal

import torch
import matplotlib.pyplot as plt
from einops import rearrange

from Code.Module.Classifier.MMCNN.MMCNNModel import MMCNNModel
from Code.Module.EEG_Encoder.TF_Encoder import Classifier
from Code.Utils.Func import data_load_denorm, data_load


@dataclass
class EnvCfg:
    data_path: str
    data_name: str

    bsz: int
    classifier_ckp: str
    classifier_ckp_tfe: str
    device: torch.device


class EEGAugEnv:
    def __init__(self, env_cfg: EnvCfg, args=None):
        self.cfg = env_cfg
        self.cfg.device = torch.device(env_cfg.device)
        self.classifier = None

        self.args = args

        self.origin_data = None
        self.step_label = None
        self.data_gen = None

        self._loadData()
        self._loadClassifier()

    def _loadClassifier(self):
        self.classifier = MMCNNModel().to(self.cfg.device)
        self.classifier.load_state_dict(torch.load(self.cfg.classifier_ckp))

    def _loadData(self):
        data_path = self.cfg.data_path
        self.data = {}
        for data_mode in ['train', 'test', 'val']:
            self.data[data_mode] = data_load(data_mode, data_path, pt=False)
        

    def reset(self, reset_mode="train", gen_mode=0, build_gen=-1)->Tuple[torch.Tensor, bool, torch.Tensor]:
        """
        reset the env into init situation
        For eeg, return a new batch pure eeg
        :param reset_mode: to indicate train, test or val
        :param gen_mode: 0 for random, 1 for order
        :param build gen: 0 for new gen, !0 for using old gen, only for gen_mode == 1
        :return:
        """
        assert reset_mode in ['train', 'test', 'val'], f'Error Env reset mode {reset_mode}'

        if self.data_gen is None or build_gen == 0:
            print(f"new data gen for {reset_mode}, with mode {gen_mode}")
            if (gen_mode == 0):
                self.data_gen = self._batchGenerator(self.data[reset_mode])
                self.data_gen = cycle(self.data_gen)
                self.train = True
            else:
                self.data_gen = self._batchGeneratorOrder(self.data[reset_mode])
                self.train = False

        indices = next(self.data_gen)
        self.step_data = torch.from_numpy(self.data[reset_mode][0][indices]).float()
        if self.step_data.dim() == 1:
            self.step_data = self.step_data.reshape(1, -1)
        self.step_label = torch.from_numpy(self.data[reset_mode][2][indices])
        self.origin_data = copy.deepcopy(self.step_data)
        self.step_data_aug = None
        self.done = False

        return self.step_data, self.done, self.step_label

    @staticmethod
    def _batchGenerator(arr_tuple):
        while True:
            indices = np.random.choice(len(arr_tuple[0]), size=1, replace=False)
            yield indices

    @staticmethod
    def _batchGeneratorOrder(arr_tuple):
        num_samples = len(arr_tuple[0])
        index = -1

        while True:
            if index <= num_samples:
                index += 1
                yield index
            else:
                raise StopIteration

    def step(self, action:np.ndarray)->Tuple[torch.Tensor, float, bool]:
        """
        doing for the eeg which didn't Done in a batch of eeg
        :param action: Gauss Kernels
        :return:
        """

        self.reward = 0

        if self.step_data_aug is not None:
            self.step_data = self.step_data_aug.reshape(1, -1)
        else:
            self.step_data = self.step_data.reshape(1, -1)

        # data_len = self.step_data.shape[-1]

        # 1. using rfft instead of fft
        #    get real data && imag data (still using abs && angle for adapte)
        self.step_data_f = torch.fft.rfft(self.step_data, dim=1)
        self.step_data_f_abs = self.step_data_f.real
        self.step_data_f_angle = self.step_data_f.imag

        # action to gauss kernels
        # in new method, the input "action" in the step() is direct sample from the distribution
        # so we need to transform the action to gauss kernels
        action:torch.Tensor = torch.tensor(action, dtype=torch.float)
        action = action.reshape(-1, 2)
        for i in range(action.shape[0]):
            action[i][0] = torch.sigmoid(action[i][0])  # mu 
            action[i][1] = torch.sigmoid(action[i][1])  # sigma

        # 2. data aug
        self._dataAug(action)

        # 3. get reward
        self._getReward(action)

        return self.step_data_aug, self.reward, self.done

    @staticmethod
    def _getWeight(origin_gsk, seg_width):
        m, s = origin_gsk[0], origin_gsk[1]
        weight = torch.linspace(0, 1, seg_width.item())
        w = 0.3989422804014327
        weight = w/s * torch.exp(-0.5 * ((weight - m) / s) ** 2)

        return weight


    def _dataAug(self, gauss_kernels, ):
        """
        1. location of the segment: using mu * data_len
        2. width of the segment: [mu * data_len - sigma * (data_len / 2 * kernel_num), mu * data_len + sigma * (data_len / 2 * 2 * kernel_num)]
        3. weight of the segment: using PDF of the Gaussian distribution (mu, sigma) to weight the segment
        """
        origin_mu = gauss_kernels[:, 0].clone()
        origin_sigma = gauss_kernels[:, 1].clone()

        symbol = torch.where(gauss_kernels[:, 1] < 0, torch.tensor(-1), torch.tensor(1))
        
        mu = gauss_kernels[:, 0] * self.step_data_f_abs.shape[-1]
        sigma = torch.abs(gauss_kernels[:, 1]) * (self.step_data_f_abs.shape[-1] / (len(gauss_kernels[:, -1]) + 4))

        mu = torch.floor(mu)
        sigma = torch.floor(sigma)

        locate_start = mu - sigma
        locate_end = mu + sigma
        locate_start = torch.clamp(locate_start, min=0, max=self.step_data_f_abs.shape[-1] - 1) 
        locate_end = torch.clamp(locate_end, min=0, max=self.step_data_f_abs.shape[-1] - 1) 
        locate_width = (locate_end - locate_start).int()


        locate = torch.stack((locate_start, locate_end), dim=1)

        assert len(locate) == len(sigma), 'size not match'
        for start_end,m, si, lw, sy in zip(locate, origin_mu, origin_sigma, locate_width, symbol):
            w = self._getWeight([m, si], lw)
            w = w * sy
            s, e = int(start_end[0].item()), int(start_end[1].item())
            assert len(w) == len(self.step_data_f_abs[0][s: e]), 'size not match'
            self.step_data_f_abs[0][s: e] = (1 + w) * self.step_data_f_abs[0][s: e] 
        
        # transform the data back to time-domain
        self.step_data_aug_f = self.step_data_f_abs + 1j * self.step_data_f_angle
        self.step_data_aug = torch.fft.irfft(self.step_data_aug_f, n = 1000)  

     

    def _getReward(self, gauss_kernels):
        step_reward = None
        logic_reward = None
        mu_reward = None
        mse_reward = None

        # 1. classifier
        self.step_data_aug = self.step_data_aug.float().to(self.cfg.device)
        pre_label = self.classifier(self.step_data_aug).cpu().detach()
        self.step_data_aug = self.step_data_aug.cpu().detach().float()


        logic_pre = torch.softmax(pre_label, dim=-1)

        # 2a. assert if predict label == real label
        pre_label = torch.argmax(pre_label, dim=-1)
        self.step_label = torch.argmax(self.step_label, dim=-1)
        self.done = torch.eq(pre_label, self.step_label).item()
        assert type(self.done) is bool, f'Error for done type {type(self.done)}'

        logic_reward = torch.abs(logic_pre[0][0] - logic_pre[0][1]).item()
        assert type(logic_reward) is float, f'Error for logic_reward type {type(logic_reward)}'
        if not torch.eq(pre_label,self.step_label).item():
            logic_reward *=-1

        # 2b. get mse in time-domain
        step_data_aug_norm = (self.step_data_aug - self.step_data_aug.min()) / (self.step_data_aug.max() - self.step_data_aug.min())
        origin_data_norm = (self.origin_data - self.origin_data.min()) / (self.origin_data.max() - self.origin_data.min())

        mse_reward = torch.nn.functional.mse_loss(input=step_data_aug_norm, target=origin_data_norm, reduction='mean').item()
        assert type(mse_reward) is float, f'Error for mse_reward type {type(mse_reward)}'

        mu = gauss_kernels[:, 0]
        mu_reward = 0
        if len(mu) != 1:
            for i in range(len(mu)):
                for j in range(i+1, len(mu)):
                    mu_diff = torch.abs(mu[i] - mu[j]).item()
                    mu_reward += mu_diff
        else:
            mu_reward = 0

        # 3. update reward new method with simple mode for (only for bsz == 1)
        if self.done:
            step_reward = 1
        else:
            step_reward = 0
   
        self.reward = step_reward + logic_reward + mu_reward + mse_reward
            

