import argparse
import os
from loguru import logger


DATASET_CONFIGS = {
    "ciao": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/ciao/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/ranking_best/best_model_ciao",
        "train_model_weights_path": "/content/drive/My Drive/Causa/ranking_best/weight/ciao",
        "log_path"                : "/content/drive/My Drive/Causa/ranking_best/logs/ciao_train.log",
        "lr"        : 0.0001,
        "epochs"    : 100,
        "K"         : 5,
        "lambda1"   : 0.1,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
        "batch_size": 128,
    },
    "dianping": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/dianping/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/ranking_best/best_model_dianping",
        "train_model_weights_path": "/content/drive/My Drive/Causa/ranking_best/weight/dianping",
        "log_path"                : "/content/drive/My Drive/Causa/ranking_best/logs/dianping_train.log",
        "lr"        : 0.001,
        "epochs"    : 50,
        "K"         : 5,
        "lambda1"   : 0.1,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
         "batch_size": 128,
    },
    "epinions": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/epinions/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/ranking_best/best_model_epinions",
        "train_model_weights_path": "/content/drive/My Drive/Causa/ranking_best/weight/epinions",
        "log_path"                : "/content/drive/My Drive/Causa/ranking_best/logs/epinions_train.log",
        "lr"        : 0.0001,
        "epochs"    : 100,
        "K"         : 5,
        "lambda1"   : 0.1,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
         "batch_size": 128,
    },
         "douban": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/douban/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/ranking_best/best_model_douban",
        "train_model_weights_path": "/content/drive/My Drive/Causa/ranking_best/weight/douban",
        "log_path"                : "/content/drive/My Drive/Causa/ranking_best/logs/douban_train.log",
        "lr"        : 0.0001,
        "epochs"    : 100,
        "K"         : 5,
        "lambda1"   : 0.1,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
        "batch_size": 128,
    },
}


def parse():
    parser = argparse.ArgumentParser(description='CADM Ranking Task: HR@K / NDCG@K')

    
    parser.add_argument('--dataset', type=str, default='ciao',
                        choices=['ciao', 'dianping', 'epinions', 'douban'])

    
    parser.add_argument('--resume', type=str, default=None)

    
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N')

    
    parser.add_argument('--batch_size',      type=int,   default=128,  metavar='N')
    parser.add_argument('--test_batch_size', type=int,   default=1000, metavar='N')
    parser.add_argument('--epochs',          type=int,   default=None, metavar='N')
    parser.add_argument('--lr',              type=float, default=None, metavar='LR')

    
    parser.add_argument('--K',          type=int,   default=None)
    parser.add_argument('--lambda1',    type=float, default=None)
    parser.add_argument('--lambda2',    type=float, default=None)
    parser.add_argument('--beta_align', type=float, default=None)

    
    parser.add_argument('--train_neg',      type=int, default=1,
                        help='训练时每个正样本对应的负样本数（默认 1:1）')
    parser.add_argument('--test_neg',       type=int, default=99,
                        help='测试时每个正样本对应的负样本数（默认 1:99）')
    parser.add_argument('--top_k',          type=int, default=10,
                        help='HR@K / NDCG@K 的截断值')
    parser.add_argument('--eval_interval',  type=int, default=5,
                        help='每隔几个 epoch 评估一次')
    parser.add_argument('--users_per_eval', type=int, default=500,
                        help='每次评估随机抽取的用户数（抽样评估加速）')

    
    parser.add_argument('--dir_data',                 type=str, default=None)
    parser.add_argument('--save_model_path',          type=str, default=None)
    parser.add_argument('--train_model_weights_path', type=str, default=None)
    
    parser.add_argument('--log_path',                 type=str, default=None)

    
    parser.add_argument('--ablation', type=str, default=None,
                        choices=[None, 'wo_mgsm', 'wo_sdsi'],
                        help='消融实验：wo_mgsm=去掉MGSM | wo_sdsi=去掉SDSI')

    args = parser.parse_args()

    
    cfg = DATASET_CONFIGS[args.dataset]
    if args.dir_data                 is None: args.dir_data                  = cfg["dir_data"]
    if args.save_model_path          is None: args.save_model_path           = cfg["save_model_path"]
    if args.train_model_weights_path is None: args.train_model_weights_path  = cfg["train_model_weights_path"]
    if args.log_path                 is None: args.log_path                  = cfg["log_path"]
    if args.lr                       is None: args.lr                        = cfg["lr"]
    if args.epochs                   is None: args.epochs                    = cfg["epochs"]
    if args.K                        is None: args.K                         = cfg["K"]
    if args.lambda1                  is None: args.lambda1                   = cfg["lambda1"]
    if args.lambda2                  is None: args.lambda2                   = cfg["lambda2"]
    if args.beta_align               is None: args.beta_align                = cfg["beta_align"]
    if args.batch_size               is None: args.batch_size                = cfg["batch_size"]  

    
    os.makedirs(args.train_model_weights_path,         exist_ok=True)
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path),        exist_ok=True)

    
    logger.add(
        args.log_path,
        rotation="50 MB",
        encoding="utf-8",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.info(f"📝 日志写入: {args.log_path}")

    return args
