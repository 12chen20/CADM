import torch
import torch.nn as nn
import torch.nn.functional as F


class SDSI(nn.Module):
    

    def __init__(self, embed_dim, K=5, lambda1=0.1, lambda2=0.1, cuda="cpu"):
        
        super(SDSI, self).__init__()

        self.embed_dim = embed_dim
        self.K         = K
        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.device    = cuda

        
        
        self.mlp_u_mask = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        
        
        self.mlp_v_mask = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        
        
        
        
        self.f_do = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, 1)
        )

        
        
        
        self.f_int = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, 1)
        )

        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    
    
    
    def complementary_mask(self, u_mixed, v_mixed):
        
        
        m_u_int = self.mlp_u_mask(u_mixed)   
        m_v_int = self.mlp_v_mask(v_mixed)   

        
        
        m_u_soc = 1.0 - m_u_int
        m_v_soc = 1.0 - m_v_int

        
        u_int = m_u_int * u_mixed
        u_soc = m_u_soc * u_mixed
        v_int = m_v_int * v_mixed
        v_soc = m_v_soc * v_mixed

        return u_int, u_soc, v_int, v_soc

    
    
    
    
    def orthogonality_loss(self, u_int, u_soc, v_int, v_soc):
        
        eps = 1e-8  

        
        cos_u = (
            torch.sum(u_int * u_soc, dim=1)
            / (torch.norm(u_int, dim=1) * torch.norm(u_soc, dim=1) + eps)
        )

        
        cos_v = (
            torch.sum(v_int * v_soc, dim=1)
            / (torch.norm(v_int, dim=1) * torch.norm(v_soc, dim=1) + eps)
        )

        
        loss_orth = torch.mean(cos_u ** 2) + torch.mean(cos_v ** 2)

        return loss_orth

    
    
    
    
    def backdoor_adjustment(self, u_int, u_soc, v_int, v_soc, labels):
        
        batch_size = u_int.size(0)

        
        z_int = torch.cat([u_int, v_int], dim=1)   

        
        z_soc = torch.cat([u_soc, v_soc], dim=1)   

        loss_bd  = 0.0
        y_do_acc = torch.zeros(batch_size).to(self.device)

        for _ in range(self.K):
            
            rand_idx = torch.randperm(batch_size).to(self.device)
            z_soc_k  = z_soc[rand_idx]   

            
            z_do_k  = torch.cat([z_int, z_soc_k], dim=1)   
            y_hat_k = self.f_do(z_do_k).squeeze(1)          

            
            loss_bd  += F.mse_loss(y_hat_k, labels)
            y_do_acc += y_hat_k.detach()   

        
        loss_bd = loss_bd  / self.K
        y_do    = y_do_acc / self.K   

        return loss_bd, y_do

    
    
    
    def forward(self, u_mixed, v_mixed, labels):
        

        
        u_int, u_soc, v_int, v_soc = self.complementary_mask(u_mixed, v_mixed)

        
        z_int     = torch.cat([u_int, v_int], dim=1)   
        y_hat_int = self.f_int(z_int).squeeze(1)        
        loss_int  = F.mse_loss(y_hat_int, labels)

        
        loss_bd, y_do = self.backdoor_adjustment(u_int, u_soc, v_int, v_soc, labels)

        
        loss_orth = self.orthogonality_loss(u_int, u_soc, v_int, v_soc)

        
        
        loss_enc = loss_int + self.lambda1 * loss_bd + self.lambda2 * loss_orth

        return loss_enc, u_int, u_soc, v_int, v_soc, y_do
