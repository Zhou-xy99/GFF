import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 

from time import time

from gnn import GNN_module
from maml import MetaLearner
from cnn import EmbeddingCNN
from cnn import myModel



def np2cuda(array):
    tensor = torch.from_numpy(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor



class GNN(myModel):
    def __init__(self, cnn_feature_size, gnn_feature_size, nway):
        super(GNN, self).__init__()

        num_inputs = cnn_feature_size + nway
        graph_conv_layer = 4  # 2
        self.gnn_obj = GNN_module(nway=nway, input_dim=num_inputs, 
            hidden_dim=gnn_feature_size, 
            num_layers=graph_conv_layer, 
            feature_type='dense')  # forward

    def forward(self, inputs):
        logits = self.gnn_obj(inputs).squeeze(-1)

        return logits
      
class gnnModel(myModel):
    def __init__(self, nway, batchsize, shots):
        super(myModel, self).__init__()
        self.batchsize = batchsize
        self.nway = nway
        self.shots = shots
        image_size = 100  # 32
        cnn_feature_size = 64
        cnn_hidden_dim = 32
        cnn_num_layers = 3

        gnn_feature_size = 32

        # self.MLT = MetaLearner(0.0001, 0.01, 32, self.nway, self.shots)
        # self.MLT.load_model()
        #
        # self.cnn_feature = self.MLT.model

        self.cnn_feature = EmbeddingCNN(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)

        # cnn_feature_ = EmbeddingCNN(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
        # self.cnn_feature = [cnn_feature_ for x in range(self.batchsize)]
        # for i in range(batchsize):
        #     self.cnn_feature[i] = self.cnn_feature[i].cuda()

        self.gnn = GNN(cnn_feature_size, gnn_feature_size, nway)

    def forward(self, data):
        # self.cnn_feature.freeze_weight()
        # for i in range(self.batchsize):
        #     self.cnn_feature[i].freeze_weight()
        [x, _, _, _, xi, label_yi, one_hot_yi, _] = data

        z = self.cnn_feature(x)
        # z = self.cnn_feature(x).repeat(1, 2)
        # print(z.size())
        # torch.Size([16, 64])

        # z = [self.cnn_feature[i](x[i, :, :, :].unsqueeze(0).repeat(2, 1, 1, 1))[0]for i in range(self.batchsize)]
        # z = torch.stack(z, dim=0)
        # # print(z.size())
        # # torch.Size([16, 64])

        # zi_s = [0 for x in range(xi.size(0))]
        # xi_sum = [xi[i, :, :, :, :] for i in range(xi.size(0))]
        # for j in range(xi.size(0)):
        #     zi_s[j] = [self.cnn_feature(xi_sum[j][i, :, :, :].unsqueeze(0)).squeeze(0) for i in range(xi.size(1))]
        #     zi_s[j] = torch.stack(zi_s[j], dim=0)
        # zi_s = torch.stack(zi_s, dim=0)
        # # print(zi_s.size())
        # # torch.Size([16, 5, 64])

        # zi_s = [0 for x in range(self.batchsize)]
        # xi_sum = [xi[i, :, :, :, :] for i in range(self.batchsize)]
        # for j in range(self.batchsize):
        #     zi_s[j] = self.cnn_feature[j](xi_sum[j])
        # zi_s = torch.stack(zi_s, dim=0)
        # # print(zi_s.size())
        # # torch.Size([16, 5, 64])

        zi = [self.cnn_feature(xi[:, i, :, :, :]) for i in range(xi.size(1))]

        zi = torch.stack(zi, dim=1)
        # print(zi.size())
        # torch.Size([16, nway*shots, 64])



        # follow the paper, concatenate the information of labels to input features
        uniform_pad = torch.FloatTensor(one_hot_yi.size(0), 1, one_hot_yi.size(2)).fill_(
            1.0/one_hot_yi.size(2))
        uniform_pad = tensor2cuda(uniform_pad)

        labels = torch.cat([uniform_pad, one_hot_yi], dim=1)
        features = torch.cat([z.unsqueeze(1), zi], dim=1)

        # print(features.size(), labels.size())
        # torch.Size([16, 6, 64]) torch.Size([16, 6, 5])
        nodes_features = torch.cat([features, labels], dim=2)

        out_logits = self.gnn(inputs=nodes_features)
        logsoft_prob = F.log_softmax(out_logits, dim=1)

        # distance_batch = [0 for x in range(self.batchsize)]
        # nodes_batch = [nodes_features[i, :, :] for i in range(self.batchsize)]
        # for b in range(self.batchsize):
        #     nodes = [nodes_batch[b][j, :] for j in range(self.nway+1)]
        #     distance = [0 for x in range(self.nway)]
        #     for i in range(self.nway):
        #         distance[i] = F.pairwise_distance(nodes[0], nodes[i+1])
        #     distance_batch[b] = torch.stack(distance, dim=0)
        # distance_batch = torch.stack(distance_batch, dim=0)
        # logsoft_prob = F.log_softmax(distance_batch, dim=1)

        return logsoft_prob

    def initialize_cnn(self, train, classes_list):
        for i in range(self.batchsize):
            self.cnn_feature[i].unfreeze_weight()
            self.cnn_feature[i] = self.MLT.get_model(train=train, classes=classes_list[i])


class Trainer():
    def __init__(self, trainer_dict):

        self.num_labels = 10

        self.args = trainer_dict['args']
        self.logger = trainer_dict['logger']

        if self.args.todo == 'train':
            self.tr_dataloader = trainer_dict['tr_dataloader']

        if self.args.model_type == 'gnn':
            Model = gnnModel
        
        self.model = Model(nway=self.args.nway, batchsize=self.args.batch_size, shots=self.args.shots)

        self.logger.info(self.model)

        self.total_iter = 0
        self.sample_size = 32

    def load_model(self, model_dir):
        self.model.load(model_dir)

        print('load model sucessfully...')

    def load_pretrain(self, model_dir):
        self.model.cnn_feature.load(model_dir)

        print('load pretrain feature sucessfully...')
    
    def model_cuda(self):
        if torch.cuda.is_available():
            self.model.cuda()

    def eval(self, dataloader, test_sample):
        self.model.eval()
        args = self.args
        iteration = int(test_sample / self.args.batch_size)

        total_loss = 0.0
        total_sample = 0
        total_correct = 0
        for i in range(iteration):
            data = dataloader.load_te_batch(batch_size=args.batch_size,
                                            nway=args.nway, num_shots=args.shots)

            # list_classes = [data[7][i, :] for i in range(args.batch_size)]
            # for i in range(args.batch_size):
            #     list_classes[i] = list_classes[i].tolist()

            data_cuda = [tensor2cuda(_data) for _data in data]
            # self.model.initialize_cnn(train=False, classes_list=list_classes)
            logsoft_prob = self.model(data_cuda)

            label = data_cuda[1]
            loss = F.nll_loss(logsoft_prob, label)

            total_loss += loss.item() * logsoft_prob.shape[0]

            pred = torch.argmax(logsoft_prob, dim=1)

            # print(pred)

            # print(torch.eq(pred, label).float().sum().item())
            # print(label)

            assert pred.shape == label.shape

            total_correct += torch.eq(pred, label).float().sum().item()
            total_sample += pred.shape[0]
        print('correct: %d / %d' % (total_correct, total_sample))
        print(total_correct)
        return total_loss / total_sample, 100.0 * total_correct / total_sample

    def train_batch(self):
        self.model.train()
        args = self.args

        data = self.tr_dataloader.load_tr_batch(batch_size=args.batch_size,
            nway=args.nway, num_shots=args.shots)

        # list_classes = [data[7][i, :] for i in range(args.batch_size)]
        # for i in range(args.batch_size):
        #     list_classes[i] = list_classes[i].tolist()

        data_cuda = [tensor2cuda(_data) for _data in data]

        self.opt.zero_grad()
        # self.model.initialize_cnn(train=True, classes_list=list_classes)
        logsoft_prob = self.model(data_cuda)  # size[16,5]

        # print('pred', torch.argmax(logsoft_prob, dim=1))
        # print('label', data[2])
        label = data_cuda[1]  # size[16]

        loss = F.nll_loss(logsoft_prob, label)  # loss在这,一个batch内16个task的loss 求和取平均值
        loss.backward()
        self.opt.step()

        return loss.item()

    # def train_batch(self):
    #     self.model.train()
    #     args = self.args
    #
    #     data = self.tr_dataloader.load_tr_batch(batch_size=args.batch_size,
    #                                             nway=args.nway, num_shots=args.shots)
    #     len_data = len(data)
    #     batch_size = args.batch_size
    #     data_split = [0 for x in range(batch_size)]
    #     temp = [0 for x in range(len(data))]
    #
    #     for i in range(batch_size):
    #         data_split[i] = [0 for x in range(len_data)]
    #     for i in range(len_data):
    #         temp[i] = data[i].split(1, dim=0)
    #     for i in range(batch_size):
    #         for j in range(len_data):
    #             if len(temp[j][i].size()) == 1:
    #                 data_split[i][j] = temp[j][i]
    #             else:
    #                 data_split[i][j] = temp[j][i].squeeze()
    #
    #     data_cuda = [0 for x in range(batch_size)]
    #     for i in range(batch_size):
    #         data_cuda[i] = [tensor2cuda(_data) for _data in data_split[i]]
    #     # self.model.change_cnn()
    #     self.opt.zero_grad()
    #     list_logsoft_prob = [0 for x in range(batch_size)]
    #     for i in range(batch_size):
    #         list_logsoft_prob[i] = self.model(data_cuda[i])  # size[16,5]
    #     logsoft_prob = torch.stack(list_logsoft_prob, dim=0)
    #     logsoft_prob_ = logsoft_prob.squeeze(1)
    #     # print('pred', torch.argmax(logsoft_prob, dim=1))
    #     # print('label', data[2])
    #     label = tensor2cuda(data[1])  # size[16]
    #
    #     loss = F.nll_loss(logsoft_prob_, label)  # loss在这,一个batch内16个task的loss 求和取平均值
    #     loss.backward()
    #     self.opt.step()
    #
    #     return loss.item()

    def train(self):
        if self.args.freeze_cnn:
            self.model.cnn_feature.freeze_weight()
            print('freeze cnn weight...')
            # for i in range(self.args.batch_size):
            #     self.model.cnn_feature[i].freeze_weight()
            # print('freeze cnn weight...')

        best_loss = 1e8
        best_acc = 0.0
        stop = 0
        eval_sample = 5000
        self.model_cuda()
        self.model_dir = os.path.join(self.args.model_folder, 'model.pth')

        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr,
            weight_decay=1e-6)
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, 
        #     weight_decay=1e-6)

        start = time()
        tr_loss_list = []
        for i in range(self.args.max_iteration):

            tr_loss = self.train_batch()
            tr_loss_list.append(tr_loss)

            if i % self.args.log_interval == 0:
                self.logger.info('iter: %d, spent: %.4f s, tr loss: %.5f' % (i, time() - start, 
                    np.mean(tr_loss_list)))
                del tr_loss_list[:]
                start = time()  

            if i % self.args.eval_interval == 0:
                va_loss, va_acc = self.eval(self.tr_dataloader, eval_sample)

                self.logger.info('================== eval ==================')
                self.logger.info('iter: %d, va loss: %.5f, va acc: %.4f %%' % (i, va_loss, va_acc))
                self.logger.info('==========================================')

                if va_loss < best_loss:
                    stop = 0
                    best_loss = va_loss
                    best_acc = va_acc
                    if self.args.save:
                        self.model.save(self.model_dir)

                stop += 1
                start = time()

                if stop > self.args.early_stop:
                    break

            self.total_iter += 1

        self.logger.info('============= best result ===============')
        self.logger.info('best loss: %.5f, best acc: %.4f %%' % (best_loss, best_acc))

    def test(self, test_data_array, te_dataloader):
        self.model_cuda()
        self.model.eval()
        start = 0
        end = 0
        args = self.args
        batch_size = args.batch_size
        pred_list = []
        while start < test_data_array.shape[0]:
            end = start + batch_size
            if end >= test_data_array.shape[0]:
                batch_size = test_data_array.shape[0] - start

            data = te_dataloader.load_te_batch(batch_size=batch_size, nway=args.nway,
                                               num_shots=args.shots)

            # print(data[5])
            # temp = data[4].squeeze(2)
            # temp_ = [temp[i, :, :, :] for i in range(2)]
            # t = [temp_[0][:, :, i] for i in range(100)]
            # # print(t[0].size())
            # np.savetxt("info1.csv", t[0].cpu().detach().numpy(), delimiter=',')

            test_x = test_data_array[start:end]

            data[0] = np2cuda(test_x)

            data_cuda = [tensor2cuda(_data) for _data in data]

            map_label2class = data[-1].cpu().numpy()

            logsoft_prob = self.model(data_cuda)
            # print(logsoft_prob)
            pred = torch.argmax(logsoft_prob, dim=1).cpu().numpy()

            pred = map_label2class[range(len(pred)), pred]

            pred_list.append(pred)

            start = end

        return np.hstack(pred_list)

    def pretrain_eval(self, loader, cnn_feature, classifier):
        total_loss = 0 
        total_sample = 0
        total_correct = 0

        with torch.no_grad():

            for j, (data, label) in enumerate(loader):
                data = tensor2cuda(data)
                label = tensor2cuda(label)
                output = classifier(cnn_feature(data))
                output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, label)

                total_loss += loss.item() * output.shape[0]

                pred = torch.argmax(output, dim=1)

                assert pred.shape == label.shape

                total_correct += torch.eq(pred, label).float().sum().item()
                total_sample += pred.shape[0]

        return total_loss / total_sample, 100.0 * total_correct / total_sample

# 原理：fine-tune，CNN-Embedding层对所有数据（每类都包含所有图片，即all shot）进行100分类，训练出初始CNN权重

    def pretrain(self, pretrain_dataset, test_dataset):
        pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, 
                batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                        batch_size=self.args.batch_size, shuffle=True)

        self.model_cuda()

        best_loss = 1e8
        self.model_dir = os.path.join(self.args.model_folder, 'pretrain_model.pth')

        cnn_feature = self.model.cnn_feature
        classifier = nn.Linear(64, self.num_labels)
        
        if torch.cuda.is_available():
            classifier.cuda()
        self.pretrain_opt = torch.optim.Adam(
            list(cnn_feature.parameters()) + list(classifier.parameters()), 
            lr=self.args.lr, 
            weight_decay=1e-6)  # 优化器

        start = time()

        for i in range(10000):
            total_tr_loss = []
            for j, (data, label) in enumerate(pretrain_loader):
                data = tensor2cuda(data)
                label = tensor2cuda(label)
                output = classifier(cnn_feature(data))

                output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, label)

                self.pretrain_opt.zero_grad()
                loss.backward()
                self.pretrain_opt.step()
                total_tr_loss.append(loss.item())

            te_loss, te_acc = self.pretrain_eval(test_loader, cnn_feature, classifier)
            self.logger.info('iter: %d, tr loss: %.5f, spent: %.4f s' % (i, np.mean(total_tr_loss), 
                time() - start))
            self.logger.info('--> eval: te loss: %.5f, te acc: %.4f %%' % (te_loss, te_acc))

            if te_loss < best_loss:
                stop = 0
                best_loss = te_loss
                if self.args.save:
                    cnn_feature.save(self.model_dir)

            stop += 1
            start = time()
        
            if stop > self.args.early_stop_pretrain:
                break



if __name__ == '__main__':
    import os
    b_s = 10
    nway = 5
    shots = 5
    batch_x = torch.rand(b_s, 3, 32, 32).cuda()
    batches_xi = [torch.rand(b_s, 3, 32, 32).cuda() for i in range(nway*shots)]

    label_x = torch.rand(b_s, nway).cuda()

    labels_yi = [torch.rand(b_s, nway).cuda() for i in range(nway*shots)]

    print('create model...')
    model = gnnModel(128, nway, b_s).cuda()
    # print(list(model.cnn_feature.parameters())[-3].shape)
    # print(len(list(model.parameters())))
    print(model([batch_x, label_x, None, None, batches_xi, labels_yi, None]).shape)
