import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from copy import deepcopy

from data import self_DataLoader
from cnn import EmbeddingCNN
from cnn import Linear_model


class BaseLearner():
    def __init__(self, learning_rate, model, classifier, nway, shot):
        self.model = deepcopy(model)
        self.classifier = deepcopy(classifier)
        self.alpha = learning_rate
        self.opt = None
        self.nway = nway
        self.shot = shot
        self.D = self_DataLoader('data', True, dataset='cifar100', nway=self.nway)
        # self.D = self_DataLoader('data', True)

    def update(self, model, classifier, learning_rate):
        self.model = deepcopy(model)
        self.classifier = deepcopy(classifier)
        self.opt = optim.SGD(list(self.model.parameters()) + list(self.classifier.parameters()), lr=learning_rate)
        # self.opt = optim.SGD(self.model.parameters(), lr=learning_rate)

    def train_task(self):
        correct = 0
        self.model = self.model.cuda()
        self.classifier = self.classifier.cuda()
        spt_x, spt_y, qry_x, qry_y = self.D.maml_task_sample(nway=self.nway, num_shots=self.shot)
        spt_x, spt_y, qry_x, qry_y = spt_x.cuda(), spt_y.cuda(), qry_x.cuda(), qry_y.cuda()
        # paras = [ele for ele in self.model.parameters()]

        ret = self.classifier(self.model(spt_x))
        loss = F.cross_entropy(ret, spt_y)
        self.opt.zero_grad()
        loss.backward()
        # grads = [ele.grad for ele in self.model.parameters()]
        self.opt.step()

        ret = self.classifier(self.model(qry_x))
        loss = F.cross_entropy(ret, qry_y)
        self.opt.zero_grad()
        loss.backward()

        correct += ret.argmax(dim=1).eq(qry_y).sum().item()

        self.model = self.model.cpu()
        self.classifier = self.classifier.cpu()
        # loss, grads, correct numbers
        return loss.item(), [ele.grad for ele in self.model.parameters()]+[ele.grad for ele in self.classifier.parameters()], correct

    def test_task(self):
        self.model = self.model.cuda()
        self.classifier = self.classifier.cuda()
        spt_x, spt_y, qry_x, qry_y = self.D.maml_task_sample(train=False, nway=self.nway, num_shots=self.shot)
        spt_x, spt_y, qry_x, qry_y = spt_x.cuda(), spt_y.cuda(), qry_x.cuda(), qry_y.cuda()

        ret = self.classifier(self.model(spt_x))
        loss = F.cross_entropy(ret, spt_y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        ret = self.classifier(self.model(qry_x))
        loss = F.cross_entropy(ret, qry_y)
        # print("Loss:", loss.item())
        correct = ret.argmax(dim=1).eq(qry_y).sum().item()
        self.model = self.model.cpu()
        self.classifier = self.classifier.cpu()
        # print("Accuracy:", correct / 5, "\n")
        return loss.item(), correct

    def cnn_task(self, classes, train):
        self.model = self.model.cuda()
        self.classifier = self.classifier.cuda()
        spt_x, spt_y = self.D.maml_cnn_task_sample(train=train, classes=classes, nway=self.nway, num_shots=self.shot)
        spt_x, spt_y = spt_x.cuda(), spt_y.cuda()

        ret = self.classifier(self.model(spt_x))
        loss = F.cross_entropy(ret, spt_y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_model(self):
        return self.model


class MetaLearner():
    def __init__(self, learning_rate, base_learning_rater,  batch_size, nway, shot):
        self.nway = nway
        self.shot = shot
        self.model_dir = os.path.join('model', '5.5_0316.pth')
        self.classifier_dir = os.path.join('model', '5.5_cla_0316.pth')
        image_size = 32  # 100
        cnn_feature_size = 64
        cnn_hidden_dim = 32  # 64
        cnn_num_layers = 3
        self.model = EmbeddingCNN(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
        self.classifier = Linear_model(self.nway)
        self.beta = learning_rate
        self.meta_batch_size = batch_size
        self.BL = BaseLearner(base_learning_rater, self.model, self.classifier, self.nway, self.shot)
        self.train_losses = list()
        self.best_loss = 1e8

    def train_one_step(self):
        grads = list()
        losses = list()
        total_correct = 0
        for batch_id in range(self.meta_batch_size):
            self.BL.update(self.model, self.classifier, self.BL.alpha)
            cur = self.BL.train_task()
            grads.append(cur[1])
            losses.append(cur[0])
            total_correct += cur[2]
        paras1 = [para for para in self.model.named_parameters()]
        paras2 = [para for para in self.classifier.named_parameters()]
        paras = paras1 + paras2
        for batch_id in range(self.meta_batch_size):
            for i in range(len(paras)):
                paras[i][1].data = paras[i][1].data - self.beta * grads[batch_id][i].data
        return sum(losses) / self.meta_batch_size, total_correct / (self.meta_batch_size * self.nway)

    def train(self, epochs):
        for meta_epoch in range(epochs):
            cur_loss, acc = self.train_one_step()
            if cur_loss < self.best_loss:
                self.best_loss = cur_loss
                self.model.save(self.model_dir)
                self.classifier.save(self.classifier_dir)
            self.train_losses.append(cur_loss)
            if (meta_epoch + 1) % 1000 == 0:
                print("Meta Training Epoch:", meta_epoch + 1)
                print("Loss:", cur_loss)
            # print("Train Accuracy:", acc)

    def test(self):
        total_correct = 0
        for batch_id in range(self.meta_batch_size):
            # print("Test task:", batch_id+1)
            self.BL.update(self.model, self.classifier, self.BL.alpha)
            cur = self.BL.test_task()
            total_correct += cur[1]

        acc = total_correct / (self.meta_batch_size * self.nway)
        print("Test Accuracy:", acc)

    def get_model(self, train, classes):
        self.BL.update(self.model, self.classifier, self.BL.alpha)
        self.BL.cnn_task(classes, train)
        model = self.BL.get_model()
        return model

    def load_model(self):
        self.model.load(self.model_dir)
        self.classifier.load(self.classifier_dir)

if __name__ == '__main__':
    meta_batch_size = 32
    alpha = 0.01
    beta = 0.0001
    nway = 5
    shot = 5

    # ML = MetaLearner(beta, alpha, meta_batch_size, nway, shot)
    # ML.load_model()
    # ML.train(80000)
    # # plt.plot(ML.train_losses)

    MLT = MetaLearner(beta, alpha, meta_batch_size, nway, shot)
    MLT.load_model()
    for i in range(20):
        MLT.test()
        # print(MLT.model.get_features())

    # lst = [5, 6, 7, 8, 9]
    # MLT.load_model()
    # print((MLT.get_model(train=True, classes=lst)))

