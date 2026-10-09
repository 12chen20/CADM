import argparse
import os
from loguru import logger






DATASET_CONFIGS = {
    "ciao": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/ciao/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/best-model/best_model_ciao",
        "train_model_weights_path": "/content/drive/My Drive/Causa/best-model/weight/ciao",
        "log_path"                : "/content/drive/My Drive/Causa/best-model/logs/ciao_train.log",
        "lr"        : 0.0001,
        "epochs"    : 100,
        "K"         : 5,
        "lambda1"   : 0.5,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
    },
    "dianping": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/dianping/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/best-model/best_model_dianping",
        "train_model_weights_path": "/content/drive/My Drive/Causa/best-model/weight/dianping",
        "log_path"                : "/content/drive/My Drive/Causa/best-model/logs/dianping_train.log",
        "lr"        : 0.001,
        "epochs"    : 50,
        "K"         : 5,
        "lambda1"   : 0.5,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
    },
    "epinions": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/epinions/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/best-model/best_model_epinions",
        "train_model_weights_path": "/content/drive/My Drive/Causa/best-model/weight/epinions",
        "log_path"                : "/content/drive/My Drive/Causa/best-model/logs/epinions_train.log",
        "lr"        : 0.0001,
        "epochs"    : 100,
        "K"         : 5,
        "lambda1"   : 0.5,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
    },
     "douban": {
        "dir_data"                : "/content/drive/My Drive/Causa/datasets/douban/mini_data/",
        "save_model_path"         : "/content/drive/My Drive/Causa/best-model/best_model_douban",
        "train_model_weights_path": "/content/drive/My Drive/Causa/best-model/weight/douban",
        "log_path"                : "/content/drive/My Drive/Causa/best-model/logs/douban_train.log",
        "lr"        : 0.0001,
        "epochs"    : 100,
        "K"         : 5,
        "lambda1"   : 0.5,
        "lambda2"   : 0.1,
        "beta_align": 0.1,
    },
}


def parse():
    
    parser = argparse.ArgumentParser(description='CADM Rating Prediction')

    
    parser.add_argument('--dataset', type=str, default='ciao',
                        choices=['ciao', 'dianping', 'epinions', 'douban'],
                        help='数据集名称，自动配置路径和超参数')

    
    parser.add_argument('--resume',       type=str,   default=None,
                        help='checkpoint 路径，用于续训')
    parser.add_argument('--start_epoch',  type=int,   default=1,
                        help='续训起始 epoch')
    parser.add_argument('--best_rmse',    type=float, default=9999.0,
                        help='续训时已知最优 RMSE')
    parser.add_argument('--best_mae',     type=float, default=9999.0,
                        help='续训时已知最优 MAE')
    parser.add_argument('--endure_count', type=int,   default=0,
                        help='续训时已累积的 endure 次数')

    
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N',
                        help='GNN 编码器输出的混合表示维度')

    
    parser.add_argument('--batch_size',      type=int,   default=128,  metavar='N')
    parser.add_argument('--test_batch_size', type=int,   default=1000, metavar='N')
    parser.add_argument('--epochs',          type=int,   default=None, metavar='N')
    parser.add_argument('--lr',              type=float, default=None, metavar='LR')

    
    parser.add_argument('--K',          type=int,   default=None,
                        help='SDSI Monte Carlo 后门调整采样次数')
    parser.add_argument('--lambda1',    type=float, default=None,
                        help='SDSI 后门调整损失 L_bd 的权重')
    parser.add_argument('--lambda2',    type=float, default=None,
                        help='SDSI 正交解耦损失 L_orth 的权重')
    parser.add_argument('--beta_align', type=float, default=None,
                        help='MGSM 因果增益对齐损失 L_align 的权重')

    
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
