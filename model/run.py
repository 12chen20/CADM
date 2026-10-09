import os
import sys

curPath     = os.path.abspath(os.path.dirname((__file__)))
rootPath    = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import warnings
warnings.filterwarnings('ignore')

import pickle
from loguru import logger

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

from para_parser import parse


from uv_aggregator import UV_Aggregator
from uv_encoder import UV_Encoder
from u_social_aggregator import U_Social_Aggregator
from u_social_encoder import U_Social_Encoder
from v_social_aggregator import V_Social_Aggregator
from v_social_encoder import V_Social_Encoder


from cadm import CADM


def data_to_list(path):
    
    df = pd.read_csv(path)
    data_u, data_v, data_r = [], [], []
    for x in df.groupby(by='user_id'):
        data_u.extend(list(x[1]['user_id']))
        data_v.extend(list(x[1]['item_id']))
        data_r.extend(list(x[1]['rating']))
    return data_u, data_v, data_r


def train(enc_u, enc_v, cadm_model, device, train_loader, optimizer, epoch):
    
    enc_u.train()
    enc_v.train()
    cadm_model.train()

    running_loss = 0.0
    running_enc  = 0.0
    running_dec  = 0.0
    total_loss   = []
    tmp_pred, target = [], []

    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        labels = labels_list.to(device)

        optimizer.zero_grad()

        
        
        
        u_enc   = enc_u(batch_nodes_u.to(device))          
        v_enc   = enc_v(batch_nodes_v.to(device))          
        u_mixed = u_enc                                     
        v_mixed = v_enc                                     

        
        loss, scores, loss_enc, loss_dec = cadm_model(u_mixed, v_mixed, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_enc  = running_enc  + loss_enc.item()  if 'running_enc'  in dir() else loss_enc.item()
        running_dec  = running_dec  + loss_dec.item()  if 'running_dec'  in dir() else loss_dec.item()
        tmp_pred.append(list(scores.detach().cpu().numpy()))
        target.append(list(labels_list.numpy()))

        if i % 100 == 0:
            expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
            mae           = mean_absolute_error(tmp_pred, target)
            tmp_pred, target = [], []

            logger.info(
                '[%d, %5d] loss: %.3f  (enc: %.3f | dec: %.3f)  rmse: %.3f  mae: %.3f'
                % (epoch, i, running_loss / 100,
                   running_enc / 100, running_dec / 100,
                   expected_rmse, mae)
            )
            total_loss.append(running_loss / 100)
            running_loss = 0.0
            running_enc  = 0.0
            running_dec  = 0.0

    logger.info('train loss: %.3f' % (sum(total_loss) / len(total_loss)))
    return total_loss


def test(enc_u, enc_v, cadm_model, device, test_loader):
    
    enc_u.eval()
    enc_v.eval()
    cadm_model.eval()

    tmp_pred, target, test_total_loss = [], [], []

    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u     = test_u.to(device)
            test_v     = test_v.to(device)
            tmp_target = tmp_target.to(device)

            
            u_enc   = enc_u(test_u)                    
            v_enc   = enc_v(test_v)                    
            u_mixed = u_enc                             
            v_mixed = v_enc                             

            
            output = cadm_model.predict(u_mixed, v_mixed)   

            
            loss, _, *_ = cadm_model(u_mixed, v_mixed, tmp_target)

            tmp_pred.append(list(output.cpu().numpy()))
            target.append(list(tmp_target.cpu().numpy()))
            test_total_loss.append(loss.item())

    tmp_pred = np.array(sum(tmp_pred, []))
    target   = np.array(sum(target,   []))

    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae           = mean_absolute_error(tmp_pred, target)

    logger.info('loss: %.3f, test rmse: %.3f, test mae: %.3f'
                % (sum(test_total_loss) / len(test_total_loss), expected_rmse, mae))
    return expected_rmse, mae, test_total_loss


def loss_plot(train_loss_list, test_loss_list, dataset):
    
    epochs = range(1, len(train_loss_list) + 1)
    plt.plot(epochs, train_loss_list, 'b', label='Train Loss')
    plt.plot(epochs, test_loss_list,  'r', label='Test Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    save_dir = '/content/drive/My Drive/Causa/best-model/logs/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset}_train_test_loss.png'), dpi=300)
    plt.show()


def main():
    
    
    args = parse()

    logger.info("=" * 50)
    logger.info(f"消融模式  : {getattr(args, 'ablation', None) or '完整模型(baseline)'}")
    logger.info(f"数据集     : {args.dataset}")
    logger.info(f"dir_data   : {args.dir_data}")
    logger.info(f"embed_dim  : {args.embed_dim}")
    logger.info(f"lr={args.lr}  epochs={args.epochs}")
    logger.info(f"K={args.K}  lambda1={args.lambda1}  "
                f"lambda2={args.lambda2}  beta_align={args.beta_align}")
    logger.info("=" * 50)

    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    dir_data = args.dir_data

    with open(dir_data + "user_purd_items.pickle",  'rb') as f: history_u_lists       = pickle.load(f)
    with open(dir_data + "item_purd_users.pickle",  'rb') as f: history_v_lists       = pickle.load(f)
    with open(dir_data + "user_item_rating.pickle", 'rb') as f: history_ur_lists      = pickle.load(f)
    with open(dir_data + "item_user_rating.pickle", 'rb') as f: history_vr_lists      = pickle.load(f)
    with open(dir_data + "user_social.pickle",      'rb') as f: user_social_adj_lists = pickle.load(f)
    with open(dir_data + "item_social.pickle",      'rb') as f: item_social_adj_lists = pickle.load(f)

    train_u, train_v, train_r = data_to_list(dir_data + 'train.csv')
    test_u,  test_v,  test_r  = data_to_list(dir_data + 'test.csv')

    ratings_list = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    
    trainset = torch.utils.data.TensorDataset(
        torch.LongTensor(train_u),
        torch.LongTensor(train_v),
        torch.FloatTensor(train_r))
    testset  = torch.utils.data.TensorDataset(
        torch.LongTensor(test_u),
        torch.LongTensor(test_v),
        torch.FloatTensor(test_r))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        testset,  batch_size=args.test_batch_size, shuffle=True)

    
    num_users   = len(history_u_lists)
    num_items   = len(history_v_lists)
    num_ratings = len(ratings_list)

    
    u2e = nn.Embedding(num_users,   args.embed_dim).to(device); nn.init.xavier_uniform_(u2e.weight)
    v2e = nn.Embedding(num_items,   args.embed_dim).to(device); nn.init.xavier_uniform_(v2e.weight)
    r2e = nn.Embedding(num_ratings, args.embed_dim).to(device)

    
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=True).to(device)
    enc_u_history = UV_Encoder(u2e, args.embed_dim, history_u_lists, history_ur_lists,
                               agg_u_history, cuda=device, uv=True).to(device)
    agg_u_social  = U_Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e,
                                        args.embed_dim, cuda=device).to(device)
    enc_u         = U_Social_Encoder(lambda nodes: enc_u_history(nodes).t(), args.embed_dim,
                                     user_social_adj_lists, agg_u_social,
                                     base_model=enc_u_history, cuda=device).to(device)

    
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=False).to(device)
    enc_v_history = UV_Encoder(v2e, args.embed_dim, history_v_lists, history_vr_lists,
                               agg_v_history, cuda=device, uv=False).to(device)
    agg_v_social  = V_Social_Aggregator(lambda nodes: enc_v_history(nodes).t(), v2e,
                                        args.embed_dim, cuda=device).to(device)
    enc_v         = V_Social_Encoder(lambda nodes: enc_v_history(nodes).t(), args.embed_dim,
                                     item_social_adj_lists, agg_v_social,
                                     base_model=enc_v_history, cuda=device).to(device)

    
    
    
    cadm_model = CADM(
        embed_dim  = args.embed_dim,
        K          = args.K,
        lambda1    = args.lambda1,
        lambda2    = args.lambda2,
        beta       = args.beta_align,
        cuda       = device,
        ablation   = getattr(args, 'ablation', None)
    ).to(device)

    
    
    optimizer = torch.optim.RMSprop(
        list(enc_u.parameters()) +
        list(enc_v.parameters()) +
        list(cadm_model.parameters()),
        lr    = args.lr,
        alpha = 0.9
    )

    
    resume_path  = getattr(args, 'resume', None)
    start_epoch  = args.start_epoch
    best_rmse    = args.best_rmse
    best_mae     = args.best_mae
    endure_count = args.endure_count

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)

        if isinstance(ckpt, dict) and 'cadm_state_dict' in ckpt:
            
            cadm_model.load_state_dict(ckpt['cadm_state_dict'])

            if 'enc_u_state_dict' in ckpt:
                enc_u.load_state_dict(ckpt['enc_u_state_dict'])
            if 'enc_v_state_dict' in ckpt:
                enc_v.load_state_dict(ckpt['enc_v_state_dict'])

            start_epoch  = ckpt.get('epoch',        0) + 1
            best_rmse    = ckpt.get('rmse',         9999.0)
            best_mae     = ckpt.get('mae',          9999.0)
            endure_count = ckpt.get('endure_count', 0)
            logger.info(f"✅ 加载完整 checkpoint: epoch={start_epoch - 1}, "
                        f"RMSE={best_rmse:.4f}, MAE={best_mae:.4f}, "
                        f"endure={endure_count}/5")
        else:
            logger.warning(f"⚠️ checkpoint 格式不兼容，从头开始训练")
    elif resume_path:
        logger.warning(f"⚠️ 权重路径不存在: {resume_path}，从头开始训练")

    
    best_model_path = f"{args.save_model_path}.pth"
    train_loss_list, test_loss_list = [], []

    logger.info('start train')

    for epoch in range(start_epoch, args.epochs + 1):

        total_loss = train(
            enc_u, enc_v, cadm_model, device, train_loader, optimizer, epoch
        )

        expected_rmse, mae, test_total_loss = test(
            enc_u, enc_v, cadm_model, device, test_loader
        )

        
        
        
        
        
        
        
        
        
        

        
        if best_rmse > expected_rmse:
            best_rmse, best_mae, endure_count = expected_rmse, mae, 0
            torch.save({
                'epoch'           : epoch,
                'cadm_state_dict' : cadm_model.state_dict(),
                'enc_u_state_dict': enc_u.state_dict(),
                'enc_v_state_dict': enc_v.state_dict(),
                'rmse'            : best_rmse,
                'mae'             : best_mae,
                'endure_count'    : 0,
            }, best_model_path)
            logger.info(f'  ✅ 新最优！RMSE={best_rmse:.4f}, MAE={best_mae:.4f} '
                        f'— 已保存至 {best_model_path}')
        else:
            endure_count += 1
            logger.info(f'  best RMSE={best_rmse:.4f}, best MAE={best_mae:.4f} '
                        f'(endure {endure_count}/5)')

        train_loss_list.append(sum(total_loss)      / len(total_loss))
        test_loss_list.append( sum(test_total_loss) / len(test_total_loss))
        logger.info("best rmse: %.4f, best mae: %.4f" % (best_rmse, best_mae))

        if endure_count >= 5:
            logger.info('Early stopping.')
            break

    
    logger.info('train loss list: {}'.format(train_loss_list))
    logger.info('test  loss list: {}'.format(test_loss_list))
    loss_plot(train_loss_list, test_loss_list, args.dataset)


if __name__ == '__main__':
    main()
