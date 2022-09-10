import os

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

###############################################################
## Vanilla CNN model, used to extract visual features
#最终将每一幅 image 处理成为一个 64 维特征矢量
class EmbeddingCNN(myModel):

    # 标准模型
    def __init__(self, image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers):
        super(EmbeddingCNN, self).__init__()
        self.features = None

        module_list = []
        dim = cnn_hidden_dim   # 32
        for i in range(cnn_num_layers):
            if i == 0:
                module_list.append(nn.Conv2d(1, dim, 3, 1, 1, bias=False))  # nn.Conv2d(1, dim, 3, 1, 1, bias=False)
                module_list.append(nn.BatchNorm2d(dim))
            else:
                module_list.append(nn.Conv2d(dim, dim*2, 3, 1, 1, bias=False))
                module_list.append(nn.BatchNorm2d(dim*2))
                dim *= 2
            module_list.append(nn.ReLU(True))
            module_list.append(nn.MaxPool2d(2))
            image_size //= 2
        module_list.append(nn.Conv2d(dim, cnn_feature_size, image_size, 1, bias=False))
        module_list.append(nn.BatchNorm2d(cnn_feature_size))
        module_list.append(nn.ReLU(True))

        self.module_list = nn.ModuleList(module_list)
        # self.out = nn.Linear(64, 5)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):

        for l in self.module_list:
            inputs = l(inputs)

        outputs = inputs.view(inputs.size(0), -1)
        return outputs

    # def forward(self, inputs):
    #
    #     for l in self.module_list:
    #         inputs = l(inputs)
    #
    #     features = inputs.view(inputs.size(0), -1)
    #     self.features = features
    #
    #     outputs = self.out(features)
    #     return outputs

    def get_features(self):
        return self.features

    def freeze_weight(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_weight(self):
        for p in self.parameters():
            p.requires_grad = True

class Linear_model(myModel):

    def __init__(self, nway):
        super(Linear_model, self).__init__()

        self.out = nn.Linear(64, nway)  # 若使用下面的embedding ，则参数为2304

    def forward(self, input):
        output = self.out(input)
        return output