import numpy as np
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def eva(y_true, y_pred, epoch=0):
    # acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    print(f"epoch {epoch}: nmi {nmi:.4f}, ari {ari:.4f}")
    return  nmi, ari

def visuization(embeding, pred, numcluster):
    embeding = embeding.detach().numpy()
    z_embeddin = TSNE(n_components=2).fit_transform(embeding)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)

    color_set = ('red', 'lime', 'blue', 'yellow', 'magenta', 'cyan', 'gray', 'brown',
                 'maroon', 'coral', 'lawngreen', 'darkviolet', 'black', 'darkorange',
                 'olivedrab', 'navy')[:numcluster]
    color_list = [color_set[int(label)] for label in pred]
    plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
    # plt.xlabel("x axis caption")
    # plt.ylabel("y axis caption")
    # plt.legend('x1') #设置图标
    plt.show()

try:
    from sklearn.manifold import TSNE; HAS_SK = True
except:
    HAS_SK = False; print('Please install sklearn for layer visualization')