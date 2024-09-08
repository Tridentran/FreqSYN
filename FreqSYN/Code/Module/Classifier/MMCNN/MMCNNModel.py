import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())

from Code.Module.Classifier.MMCNN.ConNd import CONV1d


class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, length, *args, **kwargs):
        super(ConvBlock, self).__init__()
        # k1, k2, k3 = nb_filter  # 就是通道数，懒得改名字了
        # s1, s2, s3, s4= stride
        in_channel = 16
        # kernel size的问题11111
        # print(nb_filter.shape, "Conv_block shape")

        self.conv1 = CONV1d(16, 16, length, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(16)
        self.elu = nn.ELU()

        self.conv2 = CONV1d(16, 16, length, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        self.elu = nn.ELU()

        self.conv3 = CONV1d(16, 16, length, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(16)

        self.shortcut = nn.Sequential(
            CONV1d(16, 16, 1, stride=1, padding='same'),
            nn.BatchNorm1d(16)
        )

    def forward(self, x):
        sc = self.shortcut(x)
        # print(sc.shape, "shortcut")

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape, "shape of the res_block")

        out += sc
        out = self.elu(out)
        nn.Dropout(0.8)

        return out


# inception block
class InceptionBlock(nn.Module):

    def __init__(self, in_channels, ince_length, stride):
        super(InceptionBlock, self).__init__()

        # out channel
        # k1, k2, k3, k4 = ince_filter  # 111111以此类推，所有filter都是通道数
        # kernel size
        l1, l2, l3, l4 = ince_length
        # in_channels = [3, 3, 3, 3]

        # input-->256,1000,3
        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, l1, stride=stride, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU()
        )

        self.b2 = nn.Sequential(
            nn.Conv1d(in_channels, 16, l2, stride=stride, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU()
        )

        self.b3 = nn.Sequential(
            nn.Conv1d(in_channels, 16, l3, stride=stride, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU()
        )

        self.b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=l4, stride=stride, padding=1),
            nn.Conv1d(in_channels, 16, l4, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU()
        )

    def forward(self, x):
        # print(x.shape, "input of inception")

        b1 = self.b1(x)
        # print(b1.shape, "shape of b1")

        b2 = self.b2(x)
        # print(b2.shape, "shape of b2")

        b3 = self.b3(x)
        # print(b3.shape, "shape of b3")

        b4 = self.b4(x)
        # print(b4.shape, "shape of b4")

        inception = torch.cat((b1, b2, b3, b4), dim=2)
        # print(inception.shape, "output of inception")
        return inception  # 这里的维度


# SE
class SqueezeExcitation(nn.Module):

    def __init__(self, in_chn, ratio):
        super(SqueezeExcitation, self).__init__()

        # [256, 16, 495]
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_chn, in_chn // ratio, bias=False),
            nn.ELU(),
            nn.Linear(in_chn // ratio, in_chn, bias=False),
            nn.Sigmoid()
        )

        # 原维度excitation = Reshape((1,out_dim))(excitation)

    def forward(self, x):
        # print((x.shape, "SE input"))
        b, c, _ = x.size()

        out = self.squeeze(x).view(b, c)
        # print(out.shape, "Avgpool")
        out = self.fc(out).view(b, c, 1)
        # print(out.shape, "FC")
        # out = self.f(out)
        # print(out.shape, "fllaten")
        scale = x * out.expand_as(x)

        return scale


class MMCNNModel(nn.Module):

    # parameters
    def __init__(self) -> None:
        super(MMCNNModel, self).__init__()
        self.channels = 3
        self.sample = 1000

        # self.inception_filters = [16, 16, 16, 16]
        # EEG Inception
        self.inception_kernel_length = [[5, 10, 15, 10],
                                        [40, 45, 50, 100],
                                        [60, 65, 70, 100],
                                        [80, 85, 90, 100],
                                        [160, 180, 200, 180], ]
        self.inception_stride = [2, 4, 4, 4, 16]
        self.maxPool1_size = 4
        self.maxPool1_stride = 4

        # Residual block
        self.res_block_filters = [16, 16, 16]
        self.res_block_kernel_stride = [8, 7, 7, 7, 6]

        # SE the parameter of the third part :SE block
        self.se_block_kernel_stride = 16
        self.se_ratio = 8
        self.maxPool2_size = [4, 3, 3, 3, 2]
        self.maxPool2_stride = [4, 3, 3, 3, 2]

        # self.eca_ratio = 8

        # input_tensor = input([samples, channels])
        # print("The shape of Input is.{}".format(K.int_shape(input_tensor)))
        # EIN-a kernel size= 1*5/1*10/1*15
        # InceptionBlock (self, in_channels, ince_filter, ince_length, stride):
        self.Ein_a = nn.Sequential(
            # def __init__(self, in_channels, ince_filter, ince_length, stride):
            InceptionBlock(3, self.inception_kernel_length[0], self.inception_stride[0]),
            nn.MaxPool1d(kernel_size=self.maxPool1_size, stride=self.maxPool1_stride, padding=1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.8),

            # ConvBlock (self, nb_filter, length):
            # [256, 16, 1980]) output of inception
            ConvBlock(self.res_block_kernel_stride[0]),

            SqueezeExcitation(self.se_block_kernel_stride, self.se_ratio),
            # print("The shape of se block is.{}".format(K.int_shape(x)))

            torch.nn.Flatten()
        )

        # EIN-b
        # self.Ein_b = nn.Sequential(
        #     InceptionBlock(
        #         3,
        #         self.inception_kernel_length[1],
        #         self.inception_stride[1]
        #     ),
        #     nn.MaxPool1d(kernel_size = self.maxPool1_size, stride = self.maxPool1_stride, padding = 1),
        #     nn.BatchNorm1d(16),
        #     nn.Dropout(p = 0.8),
        #     ConvBlock(self.res_block_kernel_stride[1]),
        #     SqueezeExcitation(self.se_block_kernel_stride, ratio = self.se_ratio),
        #     nn.MaxPool1d(kernel_size = self.maxPool2_size[1], stride = self.maxPool2_stride[1], padding = 1),
        #     torch.nn.Flatten()
        # )

        '''
        EIN-c
        '''
        # self.Ein_c = nn.Sequential(
        #     InceptionBlock(3, self.inception_kernel_length[2], self.inception_stride[2]),
        #     nn.MaxPool1d(kernel_size = self.maxPool1_size, stride = self.maxPool1_stride, padding = 1),
        #     nn.BatchNorm1d(16),
        #     nn.Dropout(p = 0.8),
        #     ConvBlock(self.res_block_kernel_stride[2]),
        #     SqueezeExcitation(self.se_block_kernel_stride, ratio = self.se_ratio),
        #     nn.MaxPool1d(kernel_size = self.maxPool2_size[2], stride = self.maxPool2_stride[2], padding = 1),
        #     torch.nn.Flatten()
        # )

        '''
        EIN-d
        '''
        # self.Ein_d = nn.Sequential(
        #     InceptionBlock(3, self.inception_kernel_length[3],
        #                    self.inception_stride[3]),
        #     nn.MaxPool1d(kernel_size = self.maxPool1_size, stride = self.maxPool1_stride, padding = 1),
        #     nn.BatchNorm1d(16),
        #     nn.Dropout(p = 0.8),
        #     ConvBlock(self.res_block_kernel_stride[3]),
        #     SqueezeExcitation(self.se_block_kernel_stride, ratio = self.se_ratio),
        #     # y3 = self.eca_layer(y3,self.activation,ratio= self.eca_ratio)
        #     nn.MaxPool1d(kernel_size = self.maxPool2_size[3], stride = self.maxPool2_stride[3], padding = 1),
        #     torch.nn.Flatten()
        # )

        '''
        EIN-e
        '''
        # self.Ein_e = nn.Sequential(
        #     InceptionBlock(3, self.inception_kernel_length[4],
        #                    self.inception_stride[4]),
        #     nn.MaxPool1d(kernel_size = self.maxPool1_size, stride = self.maxPool1_stride, padding = 1),
        #     nn.BatchNorm1d(16),
        #     nn.Dropout(p = 0.8),
        #     ConvBlock(self.res_block_kernel_stride[4]),
        #     # print("The shape of res(EIN-E) is.{}".format(K.int_shape(z)))
        #     SqueezeExcitation(self.se_block_kernel_stride, ratio = self.se_ratio),
        #     torch.nn.Flatten()
        # )

        self.fc = nn.Sequential(
            nn.Linear(7920, 2),
            nn.Sigmoid()
        )

        self.drop = nn.Dropout(0.8)
        self.c = nn.Conv1d(1, 3, 1)
        # output_tensor = layers.Dense(2, activation = 'sigmoid')(output_conns)
        # model = Model(input_tensor, output_tensor)

    def forward(self, x):
        # print(x.shape, "shape of Ein-a input")
        # x = x.permute(0, 2, 1)
        # print(x.shape, "shape of Ein-a input")
        x = x.unsqueeze(1)
        x = self.c(x)
        a = self.Ein_a(x)
        # print(a.shape, "shape of Ein-a")
        # b = self.Ein_b(x)
        # print(b.shape, "shape of Ein-b")
        # c = self.Ein_c(x)
        # print(c.shape, "shape of Ein-c")
        # d = self.Ein_d(x)
        # e = self.Ein_e(x)

        # output_conns = torch.cat((a, b, c, d, e), dim = 2)
        # output_conns = nn.Dropout(output_conns, self.dropout)

        output_conns = self.drop(a)
        # print(output_conns.shape, "shape of output_conns")
        output_conns = self.fc(output_conns)
        # print(output_conns.shape, "shape of MMCNN")

        return output_conns

    def get_embedding(self, x):
        x = x.unsqueeze(1)
        x = self.c(x)
        a = self.Ein_a(x)
        return a


if __name__ == "__main__":
    i = torch.ones((1, 1000))
    m = MMCNNModel()
    o = m(i)
    print(o.shape)
