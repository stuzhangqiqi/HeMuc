import numpy as np
import torch as th
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import scipy
import dgl
import random
from cos_graph import cos_acm_graph, cos_yelp_graph,cos_imdb_graph

def load_data(dataset, ratio,type_num, n, k,t,p):
    if dataset == "acm":
        data = load_acm(ratio,type_num, n, k,t,p)
    elif dataset == "yelp":
        data = load_yelp(ratio,type_num, n, k,t,p)
    elif dataset == "imdb":
        data = load_imdb(ratio,type_num, n, k,t,p)
    return data

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def load_acm(ratio,type_num, n, k,t,p):#ratio为20 40 60，每个类选择标记节点的数量 type_num:[4019,7167,60]
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
    '''label'''
    label = np.load(path + "labels.npy").astype('int32')  # 0,1,2
    label = encode_onehot(label)  # (4019,3)对上面的label进行onehot编码
    label = th.FloatTensor(label)  # (4019,3)

    '''特征视图，选择的是toP-k这么默认是5，knn是预先处理好的'''
    fpath = path+"knn/c" + str(k) + ".txt"
    # 获得txt文件
    feature_edges = np.genfromtxt(fpath, dtype=np.float32)
    fedges = np.array(list(feature_edges), dtype=np.float32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(type_num[0], type_num[0]),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    # 用余弦相似度的值代替0/1
    # fedges = np.array(list(feature_edges), dtype=np.float32).reshape(feature_edges.shape)
    # fadj = np.zeros(shape=(type_num[0], type_num[0]))
    # for i in range(len(fedges)):
    #     fadj[int(fedges[i][0])][int(fedges[i][1])] = fedges[i][-1]

    # edge_weight=fadj.data
    # edge_weight2=th.FloatTensor(edge_weight)
    # 先构图还是先归一化，按理应该是先归一化，先构图后归一化归一化后的矩阵根本没用到
    fnadj = normalize_adj(fadj + sp.eye(fadj.shape[0]))
    fadj = scipy.sparse.csr_matrix(fadj)
    fnadj = sparse_mx_to_torch_sparse_tensor(fnadj)
    gf=dgl.DGLGraph(fadj)
    gf = dgl.add_self_loop(gf)


    '''tuopu视图'''
    pasadj=np.zeros(shape=(type_num[-1],type_num[-1]),dtype=int)
    with open(path+"pa.txt") as f:
        for line in f.readlines():
            i,j=line.strip().split("\t")
            i=int(i)
            j=int(j)
            pasadj[i][j+4018]=pasadj[j+4018][i]=1

    with open(path+"ps.txt") as f:
        for line in f.readlines():
            i,j=line.strip().split("\t")
            i=int(i)
            j=int(j)
            pasadj[i][j+11185]=pasadj[j+11185][i]=1

    # 二次幂
    # print("111111111111")
    # An = matrixPow(pasadj, n)  # 数组求n次幂
    # padj = An[:4019, :4019]
    # 保存一下
    # padj4=scipy.sparse.csr_matrix(padj)
    print("222222222222")
    # scipy.sparse.save_npz(path + "/padj4.npz", padj4)
    # # scipy.sparse.save_npz(path + "/padj4.npz", padj)
    #
    # # print("333333333333333")
    padj2 = scipy.sparse.load_npz(path + 'padj2.npz')
    # padj3 = scipy.sparse.load_npz(path + 'padj3.npz')
    # padj4 = scipy.sparse.load_npz(path + 'padj4.npz')

    padj=padj2
    arrpadj = np.zeros(shape=(padj.shape[0], padj.shape[0]), dtype=float)

    for i in range(padj.shape[0]):
        for j in range(padj.shape[0]):
            arrpadj[i, j] = padj[i, j]


    for i in range(len(arrpadj)):
        for j in range(len(arrpadj)):
            if arrpadj[i][j] >=p:
                pass
            else:
                arrpadj[i][j] = 0
    padj = arrpadj
    # 稀疏矩阵
    # padj = scipy.sparse.csr_matrix(padj)
    # 去噪，用余弦相似度矩阵
    #
    #
    #
    cos_graph=cos_acm_graph(t)
    padj = padj*cos_graph

    padj = normalize_adj(padj)
    pkadj = sparse_mx_to_torch_sparse_tensor(padj)
    gp = dgl.DGLGraph(padj)
    gp = dgl.add_self_loop(gp)

    # 特征
    features_p = scipy.sparse.load_npz(path + 'p_feat.npz').todense()
    features_a = scipy.sparse.load_npz(path + 'a_feat.npz').todense()
    # features_s = scipy.sparse.load_npz(path + 'features_2.npz').todense()

    # 数据转变格式，变成tensor（高维数组）
    feat_p = th.FloatTensor(features_p)
    feat_a = th.FloatTensor(features_a)
    # feat_s = th.FloatTensor(features_s)

    # 训练验证测试集
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    # 返回的是p的邻居，PAS的特征，元路径下的P节点，pos,标签，训练集测试集验证集

    pos = sp.load_npz(path + "/pos_acm_9.npz")
    pos = sparse_mx_to_torch_sparse_tensor(pos)

    # return gp, gf, edge_weight1,edge_weight2,pkadj, fnadj, feat_p, pos, label, train, val, test
    return gp, gf, pkadj, fnadj, feat_p, pos, label, train, val, test

def load_yelp(ratio,type_num, n, k,t,p):#ratio为20 40 60，每个类选择标记节点的数量 type_num:[4019,7167,60]
    # The order of node types: 0 p 1 a 2 s
    path = "../data/yelp/"
    '''label'''
    label = np.load(path + "labels.npy").astype('int32')  # 0,1,2
    label = encode_onehot(label)  # (4019,3)对上面的label进行onehot编码
    label = th.FloatTensor(label)  # (4019,3)

    '''特征视图，选择的是toP-k这么默认是29，knn是预先处理好的'''
    fpath = path+"knn/c" + str(k) + ".txt"
    # 获得txt文件
    feature_edges = np.genfromtxt(fpath, dtype=np.float32)
    fedges = np.array(list(feature_edges), dtype=np.float32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(type_num[0], type_num[0]),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    # 用余弦相似度的值代替0/1
    # fedges = np.array(list(feature_edges), dtype=np.float32).reshape(feature_edges.shape)
    # fadj = np.zeros(shape=(type_num[0], type_num[0]))
    # for i in range(len(fedges)):
    #     fadj[int(fedges[i][0])][int(fedges[i][1])] = fedges[i][-1]

    # edge_weight=fadj.data
    # edge_weight2=th.FloatTensor(edge_weight)
    # 先构图还是先归一化，按理应该是先归一化，先构图后归一化归一化后的矩阵根本没用到
    fnadj = normalize_adj(fadj + sp.eye(fadj.shape[0]))
    fadj = scipy.sparse.csr_matrix(fadj)
    fnadj = sparse_mx_to_torch_sparse_tensor(fnadj)
    gf=dgl.DGLGraph(fadj)
    gf = dgl.add_self_loop(gf)


    '''tuopu视图'''
    # # pasadj = np.zeros(shape=(type_num[-1], type_num[-1]), dtype=int)
    # # bub = open(path + "0/b-u-b.adjlist", "r", encoding='utf-8')
    # # bub00 = [line.strip() for line in bub]
    # # bub00 = bub00[3:]
    # # bsb = open(path + "0/b-s-b.adjlist", "r", encoding='utf-8')
    # # bsb00 = [line.strip() for line in bsb]
    # # bsb00 = bsb00[3:]
    # # blb=open(path+"0/b-l-b.adjlist","r",encoding='utf-8')
    # # blb00 = [line.strip() for line in blb]
    # # blb00 = blb00[3:]
    # #
    # # # b-u
    # # for i in range(2614):
    # #     for j in bub00[i]:
    # #         if j==" ":
    # #             pass
    # #         else:
    # #             j=int(j)
    # #             pasadj[i][j+2614]+= 1
    # #             pasadj[j + 2614][i] += 1
    # # # b-s
    # # for i in range(2614):
    # #     for j in bsb00[i]:
    # #         if j==" ":
    # #             pass
    # #         else:
    # #             j=int(j)
    # #             pasadj[i][j+3900]+= 1
    # #             pasadj[j + 3900][i] += 1
    # # # b-l
    # # for i in range(2614):
    # #     for j in blb00[i]:
    # #         if j==" ":
    # #             pass
    # #         else:
    # #             j=int(j)
    # #             pasadj[i][j+3904]+= 1
    # #             pasadj[j + 3904][i] += 1
    # adjM = scipy.sparse.load_npz(path + 'adjM.npz').todense()
    # An=matrixPow(adjM,n)
    #
    # padj=An[:2614, :2614]

    # padj4 = scipy.sparse.csr_matrix(padj)
    # scipy.sparse.save_npz(path + "padj4.npz", padj4)
    # padj3 = scipy.sparse.load_npz(path + 'padj3.npz')
    # padj2 = scipy.sparse.load_npz(path + 'padj2.npz')
    padj3 = scipy.sparse.load_npz(path + 'padj3.npz')
    padj=padj3
    arrpadj = np.zeros(shape=(padj.shape[0], padj.shape[0]), dtype=float)

    for i in range(padj.shape[0]):
        for j in range(padj.shape[0]):
            arrpadj[i, j] = padj[i, j]

    print(arrpadj)

    for i in range(len(arrpadj)):
        for j in range(len(arrpadj)):
            if arrpadj[i][j] >=p:


                pass
            else:
                arrpadj[i][j] = 0
    padj = arrpadj
    # 稀疏矩阵
    # padj = scipy.sparse.csr_matrix(padj)
    # 去噪，用余弦相似度矩阵
    #
    cos_graph=cos_yelp_graph(t)
    padj = padj*cos_graph

    padj = normalize_adj(padj)
    pkadj = sparse_mx_to_torch_sparse_tensor(padj)
    gp = dgl.DGLGraph(padj)
    gp = dgl.add_self_loop(gp)

    # 特征
    features_b = scipy.sparse.load_npz(path + 'features_0.npz').todense()
    # features_a = scipy.sparse.load_npz(path + 'a_feat.npz').todense()
    # features_s = scipy.sparse.load_npz(path + 'features_2.npz').todense()

    # 数据转变格式，变成tensor（高维数组）
    feat_b = th.FloatTensor(features_b)
    # feat_a = th.FloatTensor(features_a)
    # feat_s = th.FloatTensor(features_s)

    # 训练验证测试集
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    # 返回的是p的邻居，PAS的特征，元路径下的P节点，pos,标签，训练集测试集验证集

    pos = sp.load_npz(path + "/pos/pos_yelp_9.npz")
    pos = sparse_mx_to_torch_sparse_tensor(pos)

    # return gp, gf, edge_weight1,edge_weight2,pkadj, fnadj, feat_p, pos, label, train, val, test
    return gp, gf, pkadj, fnadj, feat_b, pos, label, train, val, test

def load_imdb(ratio,type_num, n, k,t,p):#ratio为20 40 60，每个类选择标记节点的数量 type_num:[4019,7167,60]
    # The order of node types: 0 p 1 a 2 s
    path = "../data/imdb/"
    '''label'''
    label = np.load(path + "labels.npy").astype('int32')  # 0,1,2
    label = encode_onehot(label)  # (4019,3)对上面的label进行onehot编码
    label = th.FloatTensor(label)  # (4019,3)

    '''特征视图，选择的是toP-k这么默认是5，knn是预先处理好的'''
    fpath = path+"knn/c" + str(k) + ".txt"
    # 获得txt文件
    feature_edges = np.genfromtxt(fpath, dtype=np.float32)
    fedges = np.array(list(feature_edges), dtype=np.float32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(type_num[0], type_num[0]),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    # 用余弦相似度的值代替0/1
    # fedges = np.array(list(feature_edges), dtype=np.float32).reshape(feature_edges.shape)
    # fadj = np.zeros(shape=(type_num[0], type_num[0]))
    # for i in range(len(fedges)):
    #     fadj[int(fedges[i][0])][int(fedges[i][1])] = fedges[i][-1]

    # edge_weight=fadj.data
    # edge_weight2=th.FloatTensor(edge_weight)
    # 先构图还是先归一化，按理应该是先归一化，先构图后归一化归一化后的矩阵根本没用到
    fnadj = normalize_adj(fadj + sp.eye(fadj.shape[0]))
    fadj = scipy.sparse.csr_matrix(fadj)
    fnadj = sparse_mx_to_torch_sparse_tensor(fnadj)
    gf=dgl.DGLGraph(fadj)
    gf = dgl.add_self_loop(gf)


    '''tuopu视图'''
    # adjM = scipy.sparse.load_npz(path + 'adjM.npz').todense()
    # An=matrixPow(adjM,n)
    #
    # padj=An[:4278, :4278]
    #
    # padj4 = scipy.sparse.csr_matrix(padj)
    # scipy.sparse.save_npz(path + "padj4.npz", padj4)
    padj2= scipy.sparse.load_npz(path + 'padj2.npz')
    padj=padj2
    arrpadj = np.zeros(shape=(padj.shape[0], padj.shape[0]), dtype=float)

    for i in range(padj.shape[0]):
        for j in range(padj.shape[0]):
            arrpadj[i, j] = padj[i, j]

    print(arrpadj)

    for i in range(len(arrpadj)):
        for j in range(len(arrpadj)):
            if arrpadj[i][j] >=p:
                pass
            else:
                arrpadj[i][j] = 0
    padj = arrpadj
    # 稀疏矩阵

    cos_graph = cos_imdb_graph(t)
    padj = padj * cos_graph


    # padj = scipy.sparse.load_npz(path + 'padj2.npz')

    padj = normalize_adj(padj)
    pkadj = sparse_mx_to_torch_sparse_tensor(padj)
    gp = dgl.DGLGraph(padj)
    gp = dgl.add_self_loop(gp)

    # 特征
    features_m = scipy.sparse.load_npz(path + 'features_0.npz').todense()
    features_d = scipy.sparse.load_npz(path + 'features_1.npz').todense()
    # features_a = scipy.sparse.load_npz(path + 'features_2.npz').todense()

    # 数据转变格式，变成tensor（高维数组）
    feat_m = th.FloatTensor(features_m)
    feat_d = th.FloatTensor(features_d)
    # feat_a = th.FloatTensor(features_a)

    # 训练验证测试集
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    # 返回的是p的邻居，PAS的特征，元路径下的P节点，pos,标签，训练集测试集验证集

    pos = sp.load_npz(path + "/pos_imdb_10.npz")#10是最好的吗？？？？？？？？？？？？

    pos = sparse_mx_to_torch_sparse_tensor(pos)

    # return gp, gf, edge_weight1,edge_weight2,pkadj, fnadj, feat_p, pos, label, train, val, test
    return gp, gf, pkadj, fnadj, feat_m, pos, label, train, val, test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):#将scipy稀疏矩阵转换为torch稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def matrixPow(Matrix,n):
    if(type(Matrix)==list):
        Matrix=np.array(Matrix)
    if(n==1):
        return Matrix
    else:
        return np.matmul(Matrix,matrixPow(Matrix,n-1))