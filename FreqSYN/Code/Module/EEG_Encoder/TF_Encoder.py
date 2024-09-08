import torch
import torch.nn as nn
import numpy as np


class DeepSeparator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, padding=5)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=15, padding=7)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = self.act(self.conv3(x))
        x4 = self.act(self.conv4(x))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.conv_f = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU()
        )

        self.conv_t = nn.Sequential(
            # DeepSeparator(),    # b, 16, l
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
            nn.LeakyReLU(),
            DeepSeparator(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
            nn.LeakyReLU(),
            DeepSeparator(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            DeepSeparator(),
            # nn.Dropout(0.5),
            # nn.Conv1d(16, 16, 3, 2),
            # nn.LeakyReLU(),
            # nn.Conv1d(16, 128, 1),
            # nn.LeakyReLU()
        )

        self.init_conv = DeepSeparator()

        self.channel_up = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(16, 16, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv1d(16, 128, 1),
            nn.LeakyReLU()
        )

        self.channel_f = nn.Conv1d(256, 256, 1)
        # self.up_dim = nn.ConvTranspose1d(256, 256, 5, 2)
        self.up_dim = nn.Linear(500, 1000)
        self.channel_down = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 1, 1),
            nn.LeakyReLU()
        )

        self.act = nn.LeakyReLU()

    def temporal2freqence(self, x):
        x = torch.cos(x)
        x = torch.fft.rfft(x)
        x = x[:, 0: 1000 // 2]
        # x[:, int(x.shape[0] * 0.4):] = 0
        x_m = torch.unsqueeze(torch.abs(x), dim=1)
        x_p = torch.unsqueeze(torch.angle(x), dim=1)
        assert x_m.shape == x_p.shape, "x_m & x_p dim dismatch"
        return torch.cat((x_m, x_p), dim=1)

    def forward(self, x):
        x_f = self.temporal2freqence(x)
        x_f = self.conv_f(x_f)
        # print(x_f.shape)

        x_t = torch.unsqueeze(x, dim=1)
        x_t = self.init_conv(x_t)
        x_rt = x_t.clone()
        x_t = self.conv_t(x_t)
        x_t += x_rt
        x_t = self.channel_up(x_t)

        x = torch.cat((x_t, x_f), dim=1)
        x = self.act(self.channel_f(x))
        x = self.act(self.up_dim(x))
        x = self.channel_down(x)
        x = x.squeeze(1)

        return x


class TF_Encoder_Actor(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.encoder = Encoder()
        # self.mean = nn.Linear(1000, output_size)
        # self.covariance_matrix = nn.Linear(1000, output_size)

        self.mean_head = nn.Sequential(
            nn.Linear(1000, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Linear(250, 125),
            nn.Tanh(),
            nn.Linear(125, output_size),
            nn.Tanh(),
        )

        self.variance_head = nn.Sequential(
            nn.Linear(1000, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Linear(250, 125),
            nn.Tanh(),
            nn.Linear(125, output_size),
            nn.Softplus(),
        )

    def forward(self, x):
        bsz = x.shape[0] if len(x.shape) == 2 else 1
        x = self.encoder(x)
        mean = self.mean_head(x)
        variance = self.variance_head(x)
      
        return mean, variance
    
    def getDist(self, x):
        mean, variance = self.forward(x)
        return torch.distributions.Normal(mean, variance)


class TF_Encoder_Critic(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.encoder = Encoder()
        self.l = nn.Sequential(
            nn.Linear(1000, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Linear(250, 125),
            nn.Tanh(),
            nn.Linear(125, output_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.l(x)
        return x

class Classifier(nn.Module):
    def __init__(self, encoder=Encoder()):
        super().__init__()
        self.encoder = encoder
        # self.pool = nn.AvgPool1d(1000, 1)
        # self.l = nn.Linear(256, 2)
        self.l = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.pool(x)
        # x = x.squeeze(-1)
        x = self.l(x)
        return x
    
    def get_embedding(self, x):
        return self.encoder(x)



if __name__ == '__main__':
    encoder = Encoder()

    classifier = Classifier(encoder)

    i = torch.ones(512, 1000)
    o = classifier(i)
    print(o.shape)