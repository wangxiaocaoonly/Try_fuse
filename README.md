# Fuse the  Conv + BatchNorm + ReLU into one function.



1. 卷积神经网络(Convolutional Neural Networks, CNN)由Cons,Batch Normalization, ReLu等组成。
2. 通过融合三层可以减少内存访问。
3. 可以融合Conv layer和BN层是可以的，因为操作相同。

FLOPS结果在`FLOPS_res.txt`中，FUSE代码在`model.py`中。