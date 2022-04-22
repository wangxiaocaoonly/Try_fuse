#  -*-  coding:utf-8 -*-
"""
理论为：k * k * c * H * W * o

"""

from torchstat import stat
from model import *


def get_idea(model):
    k = model.conv.kernel_size[0]
    c = model.conv.out_channels
    H = model.H
    W = model.W
    o = 10
    return k * k * c * H * W * o


print('CNN模型，实际FLOPs：')
stat(CNN(), (1, 28, 28))
flops = get_idea(CNN())
print('CNN模型，理论上FLOPs：', flops)

print('*' * 100)
print('Net 模型，实际FLOPS：')
stat(Net(), (1, 28, 28))
flops = get_idea(Net())
print('Net模型，理论上FLOPs：', flops)
