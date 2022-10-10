from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
import scipy

def cos_acm_graph(t):

    path = '../data/acm/'
    features = scipy.sparse.load_npz(path+'p_feat.npz').toarray()

    cos_graph = cos(features)
    for i in range(len(cos_graph)):
        for j in range(len(cos_graph)):
            if cos_graph[i][j] >= t:
                cos_graph[i][j]=1
            else:
                cos_graph[i][j]=0
    print(cos_graph.sum(axis=1))

    cos_graph = scipy.sparse.csr_matrix(cos_graph)
    # scipy.sparse.save_npz(path+"cos_graph.npz", cos_graph)
    # print(cos_graph)
    return cos_graph

def cos_yelp_graph(t):

    path = '../data/yelp/'
    features = scipy.sparse.load_npz(path+'features_0.npz').toarray()

    cos_graph = cos(features)
    for i in range(len(cos_graph)):
        for j in range(len(cos_graph)):
            if cos_graph[i][j] >= t:
                cos_graph[i][j]=1
            else:
                cos_graph[i][j]=0
    print(cos_graph.sum(axis=1))

    cos_graph = scipy.sparse.csr_matrix(cos_graph)
    # scipy.sparse.save_npz(path+"cos_graph.npz", cos_graph)
    # print(cos_graph)
    return cos_graph

def cos_imdb_graph(t):

    path = '../data/imdb/'
    features = scipy.sparse.load_npz(path+'features_0.npz').toarray()

    cos_graph = cos(features)
    for i in range(len(cos_graph)):
        for j in range(len(cos_graph)):
            if cos_graph[i][j] >= t:
                cos_graph[i][j]=1
            else:
                cos_graph[i][j]=0
    print(cos_graph.sum(axis=1))

    cos_graph = scipy.sparse.csr_matrix(cos_graph)
    # scipy.sparse.save_npz(path+"cos_graph.npz", cos_graph)
    # print(cos_graph)
    return cos_graph