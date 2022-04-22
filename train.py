#  -*-  coding:utf-8 -*-
import torch.optim as optim
from torchvision import datasets, transforms
from model import *
from torch.utils.data import DataLoader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    cross_loss = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = cross_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss


if __name__ == "__main__":
    # 设置超参数
    batch_size = 64
    seed = 1
    epochs = 1
    lr = 0.01
    momentum = 0.5

    # 如果不指定seed值，则每次生成的随机数会因时间的差异而有所不同
    torch.manual_seed(seed)
    # 设置运行的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练数据加载
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    # 测试数据加载
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    loss_0, loss_1 = 0, 0
    # 模型定义
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        print('training model Net')
        train(model, device, train_loader, optimizer, epoch)
        loss_0 = test(model, device, test_loader)

    model = CNN().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        print('training CNN')
        train(model, device, train_loader, optimizer, epoch)
        loss_1 = test(model, device, test_loader)
    print('Loss 相差: {:.6f} '.format(
        (loss_1 - loss_0)
    ))
