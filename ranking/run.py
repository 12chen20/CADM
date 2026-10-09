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
import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import pandas as pd

from para_parser import parse


from uv_aggregator import UV_Aggregator
from uv_encoder import UV_Encoder
from u_social_aggregator import U_Social_Aggregator
from u_social_encoder import U_Social_Encoder
from v_social_aggregator import V_Social_Aggregator
from v_social_encoder import V_Social_Encoder


from cadm import CADM


from get_data import Data
from evaluation import calculate_hr_ndcg








def train(enc_u, enc_v, cadm_model, device, train_loader, optimizer, epoch):
    
    enc_u.train()
    enc_v.train()
    cadm_model.train()

    running_loss = 0.0
    running_enc  = 0.0
    running_dec  = 0.0
    running_enc  = 0.0
    running_dec  = 0.0
    total_loss   = []

    for i, (batch_nodes_u, batch_nodes_v, labels_list, _) in enumerate(train_loader):
        
        labels = labels_list.to(device)   

        optimizer.zero_grad()


        
        u_mixed = enc_u(batch_nodes_u.to(device))        
        v_mixed = enc_v(batch_nodes_v.to(device))        

        
        
        loss, _, loss_enc, loss_dec = cadm_model(u_mixed, v_mixed, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_enc  += loss_enc.item()
        running_dec  += loss_dec.item()

        if i % 100 == 0:
            avg = running_loss / (i + 1)
            logger.info(
                '[epoch %d, batch %5d] loss: %.4f  (enc: %.4f | dec: %.4f)'
                % (epoch, i, running_loss / (i + 1),
                   running_enc / (i + 1), running_dec / (i + 1)))

    avg_loss = running_loss / len(train_loader)
    logger.info(f'epoch {epoch} train loss: {avg_loss:.4f}')
    total_loss.append(avg_loss)
    return total_loss





def main():
    args = parse()

    logger.info("=" * 50)
    logger.info(f"消融模式      : {getattr(args, 'ablation', None) or '完整模型(baseline)'}")
    logger.info(f"数据集        : {args.dataset}")
    logger.info(f"dir_data      : {args.dir_data}")
    logger.info(f"embed_dim     : {args.embed_dim}  batch_size={args.batch_size}")
    logger.info(f"lr={args.lr}  epochs={args.epochs}")
    logger.info(f"K={args.K}  lambda1={args.lambda1}  lambda2={args.lambda2}  beta_align={args.beta_align}")
    logger.info(f"train_neg={args.train_neg}  test_neg={args.test_neg}  top_k={args.top_k}")
    logger.info(f"eval_interval={args.eval_interval}  users_per_eval={args.users_per_eval}")
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

    
    data_train = pd.read_csv(dir_data + 'train.csv')
    data_test  = pd.read_csv(dir_data + 'test.csv')
    data_train['x_label'] = 0
    data_test['x_label']  = 1
    data_all = pd.concat([data_train, data_test], ignore_index=True)

    data_model = Data(
        data           = data_all,
        train_num_negs = args.train_neg,
        test_num_negs  = args.test_neg,
        batch_size     = args.batch_size,
    )

    
    test_pos_samples, test_neg_samples = data_model.read_test_samples(
        path=dir_data + 'test.csv'
    )
    logger.info(f"测试集大小: {len(test_pos_samples)} 个正样本，每个配 {args.test_neg} 个负样本")

    
    num_users   = len(history_u_lists)
    num_items   = len(history_v_lists)
    num_ratings = 5   

    
    u2e = nn.Embedding(num_users,   args.embed_dim).to(device); nn.init.xavier_uniform_(u2e.weight)
    v2e = nn.Embedding(num_items,   args.embed_dim).to(device); nn.init.xavier_uniform_(v2e.weight)
    r2e = nn.Embedding(num_ratings, args.embed_dim).to(device)

    
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, args.embed_dim, history_u_lists, history_ur_lists,
                               agg_u_history, cuda=device, uv=True)
    agg_u_social  = U_Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e,
                                        args.embed_dim, cuda=device)
    enc_u         = U_Social_Encoder(lambda nodes: enc_u_history(nodes).t(), args.embed_dim,
                                     user_social_adj_lists, agg_u_social,
                                     base_model=enc_u_history, cuda=device).to(device)

    
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, args.embed_dim, history_v_lists, history_vr_lists,
                               agg_v_history, cuda=device, uv=False)
    agg_v_social  = V_Social_Aggregator(lambda nodes: enc_v_history(nodes).t(), v2e,
                                        args.embed_dim, cuda=device)
    enc_v         = V_Social_Encoder(lambda nodes: enc_v_history(nodes).t(), args.embed_dim,
                                     item_social_adj_lists, agg_v_social,
                                     base_model=enc_v_history, cuda=device).to(device)

    
    cadm_model = CADM(
        embed_dim  = args.embed_dim,
        K          = args.K,
        lambda1    = args.lambda1,
        lambda2    = args.lambda2,
        beta       = args.beta_align,
        cuda       = device
    ).to(device)

    
    optimizer = torch.optim.RMSprop(
        list(enc_u.parameters()) +
        list(enc_v.parameters()) +
        list(cadm_model.parameters()),
        lr    = args.lr,
        alpha = 0.9
    )

    
    resume_path  = getattr(args, 'resume', None)
    start_epoch  = 1
    best_hr      = 0.0
    best_ndcg    = 0.0
    endure_count = 0

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and 'cadm_state_dict' in ckpt:
            cadm_model.load_state_dict(ckpt['cadm_state_dict'])
            if 'enc_u_state_dict' in ckpt: enc_u.load_state_dict(ckpt['enc_u_state_dict'])
            if 'enc_v_state_dict' in ckpt: enc_v.load_state_dict(ckpt['enc_v_state_dict'])
            start_epoch  = ckpt.get('epoch',        0) + 1
            best_hr      = ckpt.get('hr',           0.0)
            best_ndcg    = ckpt.get('ndcg',         0.0)
            endure_count = ckpt.get('endure_count', 0)
            logger.info(f"✅ 加载 checkpoint: epoch={start_epoch-1}, "
                        f"HR={best_hr:.4f}, NDCG={best_ndcg:.4f}, endure={endure_count}/5")
        else:
            logger.warning("⚠️ checkpoint 格式不兼容，从头开始训练")
    elif resume_path:
        logger.warning(f"⚠️ 权重路径不存在: {resume_path}，从头开始训练")

    best_model_path = f"{args.save_model_path}.pth"
    logger.info('start train')

    
    
    
    for epoch in range(start_epoch, args.epochs + 1):

        
        epoch_dataset = data_model.prebuild_train_epoch()
        train_loader  = torch.utils.data.DataLoader(
            epoch_dataset,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = 2,
            pin_memory  = True,
        )

        
        total_loss = train(enc_u, enc_v, cadm_model, device, train_loader, optimizer, epoch)

        
        if epoch % args.eval_interval == 0 or epoch == args.epochs:

            
            rng = np.random.RandomState(2025 + epoch)
            all_idx = list(range(len(test_pos_samples)))
            if len(all_idx) > args.users_per_eval:
                sampled_idx = rng.choice(all_idx, size=args.users_per_eval, replace=False).tolist()
            else:
                sampled_idx = all_idx

            sampled_pos = [test_pos_samples[i] for i in sampled_idx]
            sampled_neg = [test_neg_samples[i] for i in sampled_idx]

            hits, ndcgs = calculate_hr_ndcg(
                enc_u, enc_v, cadm_model,
                sampled_pos, sampled_neg,
                k      = args.top_k,
                device = device,
            )

            hr   = sum(hits)  / len(hits)
            ndcg = sum(ndcgs) / len(ndcgs)
            logger.info(f'epoch {epoch}, loss: {total_loss[-1]:.4f}, '
                        f'HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg:.4f} '
                        f'(sampled {len(sampled_idx)} users)')

            
            torch.save({
                'epoch'           : epoch,
                'cadm_state_dict' : cadm_model.state_dict(),
                'enc_u_state_dict': enc_u.state_dict(),
                'enc_v_state_dict': enc_v.state_dict(),
                'hr'              : hr,
                'ndcg'            : ndcg,
                'endure_count'    : endure_count,
            }, f'{args.train_model_weights_path}/weights_epoch_{epoch}.pth')

            
            if hr > best_hr:
                best_hr, best_ndcg, endure_count = hr, ndcg, 0
                torch.save({
                    'epoch'           : epoch,
                    'cadm_state_dict' : cadm_model.state_dict(),
                    'enc_u_state_dict': enc_u.state_dict(),
                    'enc_v_state_dict': enc_v.state_dict(),
                    'hr'              : best_hr,
                    'ndcg'            : best_ndcg,
                    'endure_count'    : 0,
                }, best_model_path)
                logger.info(f'  ✅ 新最优！HR@{args.top_k}: {best_hr:.4f}, '
                            f'NDCG@{args.top_k}: {best_ndcg:.4f} — 已保存至 {best_model_path}')
            else:
                endure_count += 1
                logger.info(f'  best HR@{args.top_k}: {best_hr:.4f}, '
                            f'best NDCG@{args.top_k}: {best_ndcg:.4f} '
                            f'(endure {endure_count}/5)')

            if endure_count >= 5:
                logger.info('Early stopping.')
                break

        else:
            logger.info(f'epoch {epoch}, loss: {total_loss[-1]:.4f} (skip eval)')


if __name__ == '__main__':
    main()
