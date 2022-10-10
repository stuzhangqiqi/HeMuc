import numpy
import torch
from utils import load_data, set_params, evaluate, acm_params
from module import MVHGAT

import warnings
import datetime
import pickle as pkl
import os
import random


warnings.filterwarnings('ignore')
args = set_params()#args获取命令行用户输入进去的参数

# if torch.cuda.is_available(): #cuda是否可用
#     device = torch.device("cuda:" + str(args.gpu))
#     torch.cuda.set_device(args.gpu)
# else:
#     device = torch.device("cpu")

device=torch.device("cpu")
## name of intermediate document 中间文件的名称 ##
own_str = args.dataset #用own_str代替现在要访问的数据库的名字

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    gp, gf, pkadj, fnadj, feat_p, pos, label, idx_train, idx_val, idx_test  = \
        load_data(args.dataset,args.ratio, args.type_num, args.n, args.k, args.t, args.p) #初始化
    nb_classes = label.shape[-1] #3
    nfeat=feat_p.shape[-1]
    print("seed ",args.seed)
    print("Dataset: ", args.dataset,"load finish!")
    
    model = MVHGAT(nfeat, args.hidden_dim,nb_classes, args.dropout,args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    # if torch.cuda.is_available():
    #     print('Using CUDA')
    #     model.cuda()
    #     # feats = [feat.cuda() for feat in feats]
    #     # mps = [mp.cuda() for mp in mps]
    #     gp = gp.to(device)
    #     gf = gf.to(device)
    #     feat_p = feat_p.cuda()
    #     edge_weight1=edge_weight1.to(device)
    #     edge_weight2 = edge_weight2.to(device)
    #
    #     pos = pos.cuda()
    #     label = label.cuda()
    #     idx_train = [i for i in idx_train]
    #     idx_val = [i for i in idx_val]
    #     idx_test = [i for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(gp, gf, feat_p ,pos)#输入到模型中有PAS三种类型节点的特征，pos，两条元路径下P节点的同质图，邻接矩阵
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'OURS_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('OURS_'+own_str+'.pkl'))
    model.eval()
    os.remove('OURS_'+own_str+'.pkl')
    embeds = model.get_embeds(gp,gf, feat_p)
    print(embeds.shape)
    print(label.shape)
    # svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeds.detach().cpu().numpy(), label.cpu().numpy(),3)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()
