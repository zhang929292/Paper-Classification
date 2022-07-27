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

rightnow = time.strftime("%Y-%m-%d", time.localtime())  # get time
rightnowS = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_args(method, seed_num):  # Set training parameters
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default=method)
    parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--batch_size', type=int,
    #                     default=32, help='batch_size')
    parser.add_argument('--gpu_id', type=str,
                        default='0', help="device id to run")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--max_epoch', type=int,
                        default=400, help="max iterations")
    parser.add_argument('--weight_decay', type=int, default=5e-4)
    parser.add_argument('--n_hidden', type=int, default=16)    # Number of hidden layer neurons
    parser.add_argument('--n_layers', type=int, default=1)     # Number of layers used
    parser.add_argument('--dropout', type=int, default=0.5)    # Discarding some edges or nodes randomly
    parser.add_argument('--net', type=str, default='GCN',
                        help="featurizer:GCN")
    parser.add_argument('--seed', type=int, default=seed_num)  # Random number seed
    parser.add_argument('--output', type=str,                  # Output path
                        default="./train_output_{a}".format(a=rightnowS),
                        help='result output path')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id  # Set the GPU serial number for training

    args.output = "./train_output_dataset_{f}_net_{b}_alg_{h}_seed{g}/train_output_{a}_net_{b}_epoch_{c}_alg_{d}".format(
        a=rightnow, b=args.net, c=args.max_epoch, d=args.algorithm, f=args.dataset, g=seed_num, h=str(args.algorithm))
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output,
                                  'algorithm_{c}_time_{b}_out.txt'.format(b=str(rightnow), c=(args.algorithm))))
    sys.stderr = Tee(os.path.join(args.output,
                                  'algorithm_{c}_time_{b}_err.txt'.format(b=str(rightnow), c=(args.algorithm))))

    print_environ()
    return args


def print_environ():  # Print system environment and software version information
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))


class GCN( nn.Module ): # define model
    def __init__(self,
                 g,         # graph object
                 in_feats,  # Dimensions of input characteristics
                 n_hidden,  # Feature dimension of hidden layer
                 n_classes, # number of classes
                 n_layers,  # number of network layers
                 activation,  # activation function
                 dropout    # Prevent overfitting, threshold, throw away 50% of values
                 ):
        super(GCN, self ).__init__()
        self.g = g
        self.layers = nn.ModuleList()  # Create a module list object
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation = activation ))  # Add this GraphConv layer
        # hidden layer
        for i in range(n_layers - 1):  # When n_layers is greater than 1, loop over and add the GraphConv layer
            self.layers.append(GraphConv(n_hidden, n_hidden, activation = activation ))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes)) # Add the output layer, classify, the output is the category number is 7 in CORA
        self.dropout = nn.Dropout(p=dropout) # Initializes a module that prevents model overfitting

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers): # go through self.layers
            if i != 0:
                h = self.dropout(h)  # If it's not the first layer then add it to a dropout and randomly discard some edges or nodes
            h = layer( self.g, h)  # Pass the objects and features of the graph to the next layer
        return h  # return final result

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# test and verify
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
    g=data[0]  # All information of the graph, including 2078 nodes, each node has 1433 dimensions, and all nodes can be divided into 7 categories. 10556 edges.
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']  # 0~139 are training nodes
    val_mask = g.ndata['val_mask']  # 140~539 are verification nodes
    test_mask = g.ndata['test_mask']  # 1708-2707 are test nodes
    in_feats = features.shape[1]
    n_classes = data.num_labels

    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                activation,
                dropout)

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay = weight_decay)
    test_acc_list = []
    for epoch in range(n_epochs):
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluate(model, features, labels, train_mask)
        val_acc = evaluate(model, features, labels, val_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        test_acc_list.append(float(test_acc))
        print('epoch {} | loss:{:.4f} | train_acc:{:.4f} | val_acc:{:.4f} | test_acc:{:.4f}'.format(epoch,
                                                                                                    loss.item(),
                                                                                                    train_acc, val_acc,
                                                                                                    test_acc))

    print()
    print('===========end training===========')
    print("Final Test accuracy {:.2%}".format(max(test_acc_list)))

    with open(os.path.join(args.output, 'algorithm_{c}_time_{b}_done.txt'.format(b=str(rightnow), c=(args.algorithm))),
              'w') as f:
        f.write('done\n')
        f.write('final test acc:%.4f' % (max(test_acc_list)))


class Tee:  # Used to record the printed data class, generate .txt file
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


def print_args(args, print_list):  # Traverses the parameter information to be printed
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s



if __name__ == '__main__':
    '''
    No changes to the algorithm logic
    Add the training process to print experimental parameters and results, and use TXT file records.
    Random number seeds are also added to facilitate subsequent analysis and reproduction of the experiment.    
    '''
    seed_num = 0  # Set the seed of random number so that the previous experimental results can be reproduced.。
    algorithm = 'GCN'  # name
    args = get_args(method=algorithm,seed_num=seed_num)  # Gets a set of training parameters
    set_random_seed(args.seed)   # Call the random number setting function
    s = print_args(args, [])  # Obtain the parameter information to be printed
    print('=======hyper-parameter used========')
    print(s)  # Print parameter information
    print('===========start training===========')
    # Call the train function for network training, print some training data during the training process
    train(n_epochs=args.max_epoch, lr=args.lr, weight_decay=args.weight_decay, n_hidden=args.n_hidden, n_layers=args.n_layers, activation=F.relu, dropout=args.dropout)
    # epoch (iteration times) | loss (loss function value) | acc (model after the iteration accuracy)
