import argparse
import sys


argv = sys.argv
dataset = argv[1]

def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    # parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--k', type=int, default=28)

    parser.add_argument('--t', type=int, default=0.4)
    parser.add_argument('--p', type=int, default=8)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60, 11246]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.02)  # 0.01
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.7)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--k', type=int, default=29)

    parser.add_argument('--t', type=float, default=0.6)
    parser.add_argument('--p', type=int, default=25)

    args, _ = parser.parse_known_args()
    args.type_num = [2614, 1286, 4, 9, 3913]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args

def imdb_params():#Fshitu zuiyou
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.02)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.5)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[3, 8])
    parser.add_argument('--lam', type=float, default=0.5)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--k', type=int, default=150)

    parser.add_argument('--t', type=int, default=0.5)
    parser.add_argument('--p', type=int, default=6)

    args, _ = parser.parse_known_args()
    args.type_num = [4278, 2081, 5257, 11616]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "imdb":
        args = imdb_params()
    elif dataset == "yelp":
        args = yelp_params()
    return args