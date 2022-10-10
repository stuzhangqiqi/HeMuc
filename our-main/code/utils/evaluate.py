import numpy as np
import os
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
import scipy
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE; HAS_SK = True
except:
    HAS_SK = False; print('Please install sklearn for layer visualization')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()#交叉熵损失函数

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]
    # 真实labels
    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    val_true=val_lbls
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    test_true=test_lbls
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        val_nmi_sum = 0
        val_ari_sum = 0

        test_nmi_sum = 0
        test_ari_sum = 0
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)
            val_preds=preds

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            test_preds=preds
            # print(test_preds)


            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

            val_nmi = nmi_score(val_true, val_preds, average_method="arithmetic")
            val_ari = ari_score(val_true, val_preds)
            val_nmi_sum+=val_nmi
            val_ari_sum+=val_ari

            test_nmi = nmi_score(test_true, test_preds, average_method="arithmetic")
            test_ari = ari_score(test_true, test_preds)

            test_nmi_sum += test_nmi
            test_ari_sum += test_ari

            # visuization(test_embs,test_preds,nb_classes)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))
        # print("val_nmi:",val_nmi_sum/200,"val_ari",val_ari_sum/200,
        #       "test_nmi:",test_nmi_sum/200,"test_ari",test_ari_sum/200,)


    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    f.close()

def visuization(embeding, pred, numcluster):
    embeding = embeding.detach().numpy()
    z_embeddin = TSNE(n_components=2).fit_transform(embeding)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)

    color_set = ('red', 'lime', 'blue', 'yellow', 'magenta', 'cyan', 'gray', 'brown',
                 'maroon', 'coral', 'lawngreen', 'darkviolet', 'black', 'darkorange',
                 'olivedrab', 'navy')[:numcluster]
    color_list = [color_set[int(label)] for label in pred]
    plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list)
    # plt.xlabel("x axis caption")
    # plt.ylabel("y axis caption")
    # plt.legend('x1') #设置图标
    plt.show()


