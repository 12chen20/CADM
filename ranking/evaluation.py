import math
import heapq
import torch


def calculate_hr_ndcg(enc_u, enc_v, cadm_model, test_ratings, test_negatives,
                      k, device, batch_size=512):
    
    enc_u.eval()
    enc_v.eval()
    cadm_model.eval()

    
    
    
    num_users = enc_u.embed_dim  
    
    max_uid = max(r[0] for r in test_ratings)
    max_vid = max(
        max(r[1] for r in test_ratings),
        max(n for negs in test_negatives for n in negs)
    )
    num_users = max_uid + 1
    num_items = max_vid + 1

    CHUNK = 256  

    all_user_emb = []
    with torch.no_grad():
        for s in range(0, num_users, CHUNK):
            end  = min(s + CHUNK, num_users)
            ids  = torch.arange(s, end, dtype=torch.long).to(device)
            
            emb  = enc_u(ids)
            all_user_emb.append(emb.cpu())
    all_user_emb = torch.cat(all_user_emb, dim=0).to(device)  

    
    all_item_emb = []
    with torch.no_grad():
        for s in range(0, num_items, CHUNK):
            end  = min(s + CHUNK, num_items)
            ids  = torch.arange(s, end, dtype=torch.long).to(device)
            emb  = enc_v(ids)
            all_item_emb.append(emb.cpu())
    all_item_emb = torch.cat(all_item_emb, dim=0).to(device)  

    
    all_users      = []   
    all_items      = []   
    sample_lengths = []   
    pos_items      = []   

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        u_id   = rating[0]                            
        i_id   = rating[1]                            
        items  = test_negatives[idx][:] + [i_id]      

        n = len(items)
        all_users.extend([u_id] * n)                  
        all_items.extend(items)                        
        sample_lengths.append(n)
        pos_items.append(i_id)

    all_users = torch.tensor(all_users, dtype=torch.long)   
    all_items = torch.tensor(all_items, dtype=torch.long)   

    
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(all_users), batch_size):
            end   = start + batch_size
            u_ids = all_users[start:end].to(device)   
            v_ids = all_items[start:end].to(device)   

            u_emb = all_user_emb[u_ids]               
            v_emb = all_item_emb[v_ids]               

            
            preds = cadm_model.predict(u_emb, v_emb)  
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0).tolist()  

    
    hits, ndcgs = [], []
    cursor = 0  

    for idx, n in enumerate(sample_lengths):
        preds_slice = all_preds[cursor: cursor + n]              
        items_slice = test_negatives[idx][:] + [pos_items[idx]]  
        cursor += n

        
        map_item_score = {items_slice[i]: preds_slice[i] for i in range(n)}
        ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)

        hits.append(get_hit_ratio(ranklist, pos_items[idx]))
        ndcgs.append(get_ndcg(ranklist, pos_items[idx]))

    return hits, ndcgs


def get_hit_ratio(ranklist, item):
    
    return 1 if item in ranklist else 0


def get_ndcg(ranklist, item):
    
    for i, each in enumerate(ranklist):
        if each == item:
            return math.log(2) / math.log(i + 2)
    return 0
