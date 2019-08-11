# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torch.utils.tensorboard import SummaryWriter


class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(x5))
        # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(x5))
        # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)
        # size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv2(score))
        # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(self.relu(self.deconv1(x5)))
        # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = score + x4
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = score + x3
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = score + x2
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # *的意义：https://blog.csdn.net/qinyilang/article/details/5484415
    return nn.Sequential(*layers)  # list没有前向传播，使用Sequential进行前向传播


if __name__ == "__main__":
    # batch_size, n_class, h, w = 10, 20, 160, 160
    batch_size, n_class, h, w = 10, 16, 256, 256
    # test output size
    vgg_model = VGGNet(requires_grad=True)
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, 256, 256))
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = vgg_model(input)
    assert output['x5'].size() == torch.Size([batch_size, 512, 8, 8])

    # h, w = 100, 100
    # FCN中经过VGG的5个2倍降采样后256*256变为8*8，再经过5个2倍上采样变为256*256，回到原始尺寸
    # 如果换成100*100则经过VGG的5个降采样变为3*3,再经过5个2倍上采样变为96*96,不能回到原尺寸
    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])

    print("Pass size check")

    # test a random batch, loss should decrease
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    y = torch.autograd.Variable(torch.randn(
        batch_size, n_class, h, w), requires_grad=False)
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(input)
        output = torch.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        print("iter{}, loss {}".format(iter, loss.data.item()))
        optimizer.step()
