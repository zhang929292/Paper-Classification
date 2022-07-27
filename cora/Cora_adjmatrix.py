# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : learning coding
 Version      : 1.0
 Author       : marzsccc
 Date         : 2022-05-03 11:15:29
 LastEditors  : marzsccc
 LastEditTime : 2022-05-03 13:04:36
 FilePath     : \\undefinede:\\work3\\gnn\\dgl_version\\cora\\Cora_adjmatrix.py
 Copyright (C) 2022 marzsccc. All rights reserved.
'''
import pandas as pd
import numpy as np

''' View data dimensions'''
# raw_data = pd.read_csv('cora.content', sep='\t', header=None)
# print("content shape: ", raw_data.shape)
#
# raw_data_cites = pd.read_csv('cora.cites', sep='\t', header=None)
# print("cites shape: ", raw_data_cites.shape)


''' Extract feature vectors and labels'''
# raw_data = pd.read_csv('cora.content', sep='\t', header=None)
# print("content shape: ", raw_data.shape)
#
# features = raw_data.iloc[:,1:-1]
# print("features shape: ", features.shape)
#
# # one-hot encoding
# labels = pd.get_dummies(raw_data[1434])
# print("\n----head(3) one-hot label----")
# print(labels.head(3))


'''Build an adjacency matrix'''
raw_data = pd.read_csv('cora.content', sep='\t', header=None)
num_nodes = raw_data.shape[0]

# 将节点重新编号为[0, 2707]
new_id = list(raw_data.index)
id = list(raw_data[0])
c = zip(id, new_id)
map = dict(c)

raw_data_cites = pd.read_csv('cora.cites', sep='\t', header=None)

# 根据节点个数定义矩阵维度
matrix = np.zeros((num_nodes, num_nodes))

# 根据边构建矩阵
for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
    x = map[i]
    y = map[j]
    matrix[x][y] = matrix[y][x] = 1   # 无向图：有引用关系的样本点之间取1

# 查看邻接矩阵的元素
print(matrix)

# 将matrix保存到txt中
np.savetxt('cora_adjmatrix.txt', matrix, fmt='%d')
# 将matrix保存到csv中
np.savetxt('cora_adjmatrix.csv', matrix, fmt='%d', delimiter=',')
#将matrix保存到matlab中
np.savetxt('cora_adjmatrix.mat', matrix, fmt='%d', delimiter=',')
