import sys
import os
from datetime import time
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import CoraGraphDataset
from numpy import random
import numpy as np
import time

from torch import autograd

rightnow = time.strftime("%Y-%m-%d", time.localtime())
rightnowS = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_args(method, seed_num):
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default=method)
    parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--batch_size', type=int,
    #                     default=32, help='batch_size')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--max_epoch', type=int,
                        default=20, help="max iterations")  # ----------------------

    parser.add_argument('--weight_decay', type=int, default=5e-4)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--net', type=str, default='GCN',
                        help="featurizer: vgg16, resnet50, GCN")
    parser.add_argument('--seed', type=int, default=seed_num)
    parser.add_argument('--output', type=str,
                        default="./train_output_{a}".format(a=rightnowS),
                        help='result output path')

    parser.add_argument('--sd_reg', type=int, default=0.1)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id

    args.output = "./train_output_dataset_{f}_net_{b}_alg_{h}_seed{g}/train_output_{a}_net_{b}_epoch_{c}_alg_{d}".format(
        a=rightnow, b=args.net, c=args.max_epoch, d=args.algorithm, f=args.dataset, g=seed_num, h=str(args.algorithm))
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output,
                                  'algorithm_{c}_time_{b}_out.txt'.format(b=str(rightnow), c=(args.algorithm))))
    sys.stderr = Tee(os.path.join(args.output,
                                  'algorithm_{c}_time_{b}_err.txt'.format(b=str(rightnow), c=(args.algorithm))))

    print_environ()
    return args


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))



class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


class GCN( nn.Module ):
    def __init__(self,
                 g,  # graph object
                 in_feats,  # Dimensions of input traits
                 n_hidden,  # Feature dimension of hidden layer
                 n_classes,  # number of classes
                 n_layers,  # number of network layers
                 activation,  # activation function
                 dropout  # dropout coefficient
                 ):
        super(GCN, self ).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv( in_feats, n_hidden, activation = activation ))
        # hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation = activation ))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)




    def forward(self, features):
        h = features
        for i, layer in enumerate( self.layers ):
            if i != 0:
                h = self.dropout( h )
            h = layer( self.g, h )
        return h

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(n_epochs=100, lr=1e-2, weight_decay=5e-4, n_hidden=16, n_layers=1, activation=F.relu , dropout=0.5,):
    data = CoraGraphDataset()
    g=data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels

    model = GCN(g,  # graph object of DGL
                in_feats,  # Dimensions of input traits
                n_hidden,  # Feature dimension of hidden layer
                n_classes,  # number of classes
                n_layers,  # number of network layers
                activation,  # activation function
                dropout)  # dropout coefficient
    print('===========model structure===========')
    print(model._modules)
    # print(model._modules.items())
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    print('===========start training===========')
    total_loss = 0
    test_acc_list = []
    for epoch in range(n_epochs):
        # When the epoch cycle is between 0.7~0.9 times of the total epoch,
        # adjust the learning rate to 0.1 times before (lr=lr*0.1)
        if epoch in [int(args.max_epoch * 0.7), int(args.max_epoch * 0.9)]:
            print('manually decrease lr')
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=lr*0.1,
                                         weight_decay=weight_decay)
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        bias_criterion = GeneralizedCELoss(q=0.7)
        loss_gce = bias_criterion(logits, labels)
        total_loss = loss + 0.1 * loss_gce.mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_acc = evaluate(model, features, labels, train_mask)
        val_acc = evaluate(model, features, labels, val_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        test_acc_list.append(float(test_acc))
        print('epoch {} | loss:{:.4f} | train_acc:{:.4f} | val_acc:{:.4f} | test_acc:{:.4f}'.format(epoch,
                                                                                                    total_loss.item(),
                                                                                                    train_acc, val_acc,
                                                                                                    test_acc))

    print()
    print('===========end training===========')
    print("Final Test accuracy {:.2%}".format(max(test_acc_list)))

    with open(os.path.join(args.output, 'algorithm_{c}_time_{b}_done.txt'.format(b=str(rightnow), c=(args.algorithm))),
              'w') as f:
        f.write('done\n')
        f.write('final test acc:%.4f' % (max(test_acc_list)))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def print_args(args, print_list):
    s = ""
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s



if __name__ == '__main__':
    '''
    loss Adding gce loss + dynamic decay learning rate but it is useless, the effect is a little lower than improvement 3
    '''
    seed_num = 0
    algorithm = 'GCN3_1'
    args = get_args(method=algorithm,seed_num=seed_num)
    set_random_seed(args.seed)
    # Set the random number seed so that the results of each training run are the same.
    # The principle can be simply understood as: every time the program is run, the variables of the model are set to the same initialization parameters, so the result of each run after a specific operation is the same. This can reproduce the previous experimental results.
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)

    train(n_epochs=args.max_epoch, lr=args.lr, weight_decay=args.weight_decay, n_hidden=args.n_hidden, n_layers=args.n_layers, activation=F.relu, dropout=args.dropout)  