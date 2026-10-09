import torch
import torch.nn as nn
import torch.nn.functional as F

from sdsi import SDSI
from mgsm import MGSM


class CADM(nn.Module):
    def __init__(
        self,
        embed_dim,
        K         = 5,
        lambda1   = 0.1,
        lambda2   = 0.1,
        beta      = 0.1,
        cuda      = "cpu",
        ablation  = None      
    ):
        super(CADM, self).__init__()

        self.device   = cuda
        self.ablation = ablation
        self.embed_dim = embed_dim

        
        self.sdsi = SDSI(
            embed_dim = embed_dim,
            K         = K,
            lambda1   = lambda1,
            lambda2   = lambda2,
            cuda      = cuda
        )

        
        self.mgsm = MGSM(
            embed_dim = embed_dim,
            beta      = beta,
            cuda      = cuda
        )

        
        
        self.f_direct = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, u_mixed, v_mixed, labels):
        
        device = self.device

        
        
        
        if self.ablation == "wo_sdsi":
            
            u_int = u_mixed
            v_int = v_mixed
            u_soc = torch.zeros_like(u_mixed)
            v_soc = torch.zeros_like(v_mixed)

            
            z_direct = torch.cat([u_int, v_int], dim=1)
            y_do = self.f_direct(z_direct).squeeze(1).detach()

            loss_enc = torch.tensor(0.0, device=device)

            
            loss_dec, y_hat = self.mgsm(u_int, u_soc, v_int, v_soc, y_do, labels)

            loss = loss_dec
            return loss, y_hat, loss_enc, loss_dec

        
        
        
        loss_enc, u_int, u_soc, v_int, v_soc, y_do = self.sdsi(
            u_mixed, v_mixed, labels
        )

        
        
        
        if self.ablation == "wo_mgsm":
            
            z_int = torch.cat([u_int, v_int], dim=1)
            z_soc = torch.cat([u_soc, v_soc], dim=1)
            y_hat_raw = self.sdsi.f_do(
                torch.cat([z_int, z_soc], dim=1)
            ).squeeze(1)

            loss_rec = F.mse_loss(y_hat_raw, labels.float())
            loss_dec = loss_rec

            loss  = loss_enc + loss_dec
            y_hat = y_hat_raw
            return loss, y_hat, loss_enc, loss_dec

        
        
        
        loss_dec, y_hat = self.mgsm(u_int, u_soc, v_int, v_soc, y_do, labels)

        loss = loss_enc + loss_dec
        return loss, y_hat, loss_enc, loss_dec

    @torch.no_grad()
    def predict(self, u_emb, v_emb):
        
        device = self.device
        dummy_labels = torch.zeros(u_emb.size(0), device=device)

        if self.ablation == "wo_sdsi":
            z = torch.cat([u_emb, v_emb], dim=1)
            return self.f_direct(z).squeeze(1)

        if self.ablation == "wo_mgsm":
            _, u_int, u_soc, v_int, v_soc, _ = self.sdsi(u_emb, v_emb, dummy_labels)
            z_int = torch.cat([u_int, v_int], dim=1)
            z_soc = torch.cat([u_soc, v_soc], dim=1)
            return self.sdsi.f_do(
                torch.cat([z_int, z_soc], dim=1)
            ).squeeze(1)

        
        _, u_int, u_soc, v_int, v_soc, y_do = self.sdsi(u_emb, v_emb, dummy_labels)
        _, y_hat = self.mgsm(u_int, u_soc, v_int, v_soc, y_do, dummy_labels)
        return y_hat

