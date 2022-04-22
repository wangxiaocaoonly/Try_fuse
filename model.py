#  -*-  coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):

    def __init__(self, num_classes=10, batch_size=64):
        super(CNN, self).__init__()
        # 定义参数
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.in_channels = 1
        self.out_channels = 16
        self.kernel_size = (5, 5)
        # 计算
        self.H = 28
        self.W = 28
        # 定义模型
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,  # 要得到几多少个特征图
            kernel_size=self.kernel_size,  # 卷积核大小
            stride=(1, 1),  # 步长
            padding=2
        )
        self.bn = nn.BatchNorm2d(16)
        self.relu=nn.ReLU()
        self.fc = nn.Linear(16 * 28 * 28, self.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(-1, 16 * 28 * 28)
        x = self.relu(x)
        x = self.fc(x)
        return x

    def backward(self, grad_output):
        return grad_output, None


class Net(nn.Module):

    def __init__(self, num_classes=10, batch_size=64):
        super(Net, self).__init__()
        # 定义参数
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.in_channels = 1
        self.out_channels = 16
        self.kernel_size = (5, 5)
        # 计算用
        self.H = 24
        self.W = 24
        # 定义模型
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,  # 要得到几多少个特征图
            kernel_size=self.kernel_size,  # 卷积核大小
            stride=(1, 1),  # 步长
            padding=2
        )
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 24 * 24, self.num_classes)

    def fold_bn(self, mean, std):
        """
            conv+bn
        :param mean: 平均值
        :param std: 标准差
        :return:
        """
        tmp = self.bn.weight / std

        weight = self.conv.weight * tmp.view(self.conv.out_channels, 1, 1, 1)
        bias = tmp * self.conv.bias - tmp * mean + self.bn.bias

        return weight, bias

    def forward(self, x):
        y = F.conv2d(x, self.conv.weight, self.conv.bias,
                     stride=self.conv.stride)
        y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW

        y = y.contiguous().view(self.conv.out_channels, -1)  # CNHW -> C,NHW

        mean = y.mean(1).detach()
        variance = y.var(1).detach()
        self.bn.running_mean = self.bn.momentum * self.bn.running_mean + (1 - self.bn.momentum) * mean
        self.bn.running_var = self.bn.momentum * self.bn.running_var + (1 - self.bn.momentum) * variance
        # 标准差
        std = torch.sqrt(variance + self.bn.eps)
        # 合并bn
        weight, bias = self.fold_bn(mean, std)
        x = F.conv2d(x, weight, bias, stride=self.conv.stride)
        # 合并relu
        x = F.relu(x)
        x = x.view(-1, 16 * 24 * 24)
        x = self.fc(x)
        return x

    def backward(self, grad_output):
        return grad_output, None
