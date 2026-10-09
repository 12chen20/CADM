import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")  


class Data(nn.Module):
    
    def __init__(self, data, train_num_negs, test_num_negs, batch_size):
        
        super(Data, self).__init__()
        self.df_all   = data                                              
        self.df_train = data[data["x_label"] == 0].reset_index(drop=True) 
        self.df_test  = data[data["x_label"] == 1].reset_index(drop=True) 
        self.train_num_negs = train_num_negs  
        self.test_num_negs  = test_num_negs   
        self.batch_size     = batch_size      

        
        self._num_items = int(data["item_id"].max()) + 1

        
        
        self._user_interacted = (
            self.df_all.groupby("user_id")["item_id"]
            .apply(set).to_dict()
        )

        
        self._train_u = self.df_train["user_id"].values.astype(np.int64)   
        self._train_v = self.df_train["item_id"].values.astype(np.int64)   
        self._train_f = np.zeros(len(self.df_train), dtype=np.float32)     

    
    def prebuild_train_epoch(self):
        
        N       = len(self._train_u)   
        K       = self.train_num_negs  
        n_items = self._num_items      

        
        pos_u = self._train_u                              
        pos_v = self._train_v                              
        pos_l = np.ones(N,  dtype=np.float32)              
        pos_f = self._train_f                              

        
        neg_u = np.repeat(pos_u, K)                        
        neg_v = np.random.randint(0, n_items, size=N * K).astype(np.int64)  
        neg_l = np.zeros(N * K, dtype=np.float32)          
        neg_f = np.zeros(N * K, dtype=np.float32)          

        
        
        for _ in range(10):
            
            mask = np.array([
                neg_v[i] in self._user_interacted.get(int(neg_u[i]), set())
                for i in range(len(neg_u))
            ], dtype=bool)
            if mask.sum() == 0:   
                break
            
            neg_v[mask] = np.random.randint(0, n_items, size=int(mask.sum()))
        

        
        all_u = np.concatenate([pos_u, neg_u])  
        all_v = np.concatenate([pos_v, neg_v])  
        all_l = np.concatenate([pos_l, neg_l])  
        all_f = np.concatenate([pos_f, neg_f])  

        
        idx = np.random.permutation(len(all_u))  
        return torch.utils.data.TensorDataset(
            torch.from_numpy(all_u[idx]),  
            torch.from_numpy(all_v[idx]),  
            torch.from_numpy(all_l[idx]),  
            torch.from_numpy(all_f[idx]),  
        )

    
    def generate_train_instances(self, batch_u, batch_v, batch_f):
        
        num_items = self._num_items
        uids, iids, ratings, f = [], [], [], []
        for i in range(len(batch_u)):
            u_id     = batch_u[i]
            pos_i_id = batch_v[i]
            uids.append(u_id);    iids.append(pos_i_id)   
            f.append(batch_f[i]); ratings.append(1)        

            interacted = self._user_interacted.get(u_id, set())  
            for _ in range(self.train_num_negs):
                j = np.random.randint(num_items)
                while j in interacted:                      
                    j = np.random.randint(num_items)
                uids.append(u_id); iids.append(j)
                ratings.append(0); f.append(0)             
        return uids, iids, ratings, f

    
    def read_test_samples(self, path, cache_path=None):
        
        
        if cache_path is None:
            cache_path = path.replace(".csv", f"_testneg{self.test_num_negs}.pkl")

        
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                pos_samples, neg_samples = pickle.load(f)
            print(f"✅ 从缓存加载测试集: {cache_path}  ({len(pos_samples)} samples)")
            return pos_samples, neg_samples

        
        if path.endswith(".csv"):
            pos_samples, neg_samples = self._read_test_samples_from_csv()   
        else:
            pos_samples, neg_samples = self._read_test_samples_from_txt(path)  

        
        with open(cache_path, "wb") as f:
            pickle.dump((pos_samples, neg_samples), f)
        print(f"✅ 测试集已缓存至: {cache_path}")
        return pos_samples, neg_samples

    def _read_test_samples_from_txt(self, path):
        
        pos_samples, neg_samples = [], []
        with open(path) as f:
            for line in f:
                
                each_content = list(map(int, line.strip().split(" ")[:101]))
                pos_samples.append([each_content[0], each_content[1], each_content[2]])  
                neg_samples.append(each_content[3:101])  
        return pos_samples, neg_samples

    def _read_test_samples_from_csv(self):
        
        num_items = self._num_items
        pos_samples, neg_samples = [], []
        total = len(self.df_test)

        for idx, row in self.df_test.iterrows():
            u_id = int(row["user_id"])
            i_id = int(row["item_id"])
            pos_samples.append([u_id, i_id])  

            interacted = self._user_interacted.get(u_id, set())  
            each_neg = []
            
            while len(each_neg) < self.test_num_negs:
                candidates = np.random.randint(0, num_items, size=self.test_num_negs * 3)
                for c in candidates:
                    
                    if c not in interacted and c not in each_neg:
                        each_neg.append(int(c))
                    if len(each_neg) == self.test_num_negs:
                        break
            neg_samples.append(each_neg)

            
            if (idx + 1) % 1000 == 0:
                print(f"  [test sampling] {idx+1}/{total}", flush=True)

        return pos_samples, neg_samples
