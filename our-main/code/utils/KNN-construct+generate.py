from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
import scipy
from sklearn.metrics import pairwise_distances as pair

# def construct_graph(dataset, features, topk):
def construct_graph(topk):
    f = '../../data/imdb/features_0.npz'
    fw=open('../../data/acm--now/knn/tmp.txt','w')
    features=scipy.sparse.load_npz(f).toarray()

    # # f = open(feature0, 'w')
    # ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):] #输出前top+1
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                fw.write('{} {} {}\n'.format(int(i), int(vv), dist[i][vv]))
    fw.close()


def generate_knn():
    for topk in range(5, 36):
        data = np.load('../../data/acm--now/p_feat.npz')
        print(data)
        construct_graph(topk)
        f1 = open('../../data/acm--now/knn/tmp.txt','r')
        f2 = open('../../data/acm--now/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end, w= line.strip('\n').split(' ')
            # if int(start) < int(end):
            f2.write('{} {} {}\n'.format(start, end,w))
        f2.close()

if __name__ == "__main__":
    generate_knn()