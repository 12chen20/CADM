import torch
import torch.nn as nn
import torch.nn.functional as F


class MGSM(nn.Module):
    

    def __init__(self, embed_dim, beta=0.1, cuda="cpu"):
        
        super(MGSM, self).__init__()

        self.embed_dim = embed_dim
        self.beta      = beta
        self.device    = cuda

        
        
        
        self.f_R = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, 1)
        )

        
        
        
        
        
        self.gate_u = nn.Linear(embed_dim * 4, 1)

        
        
        
        
        self.gate_v = nn.Linear(embed_dim * 4, 1)

        
        
        
        
        self.f_P = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
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

    
    
    
    
    def causal_gain_estimation(self, u_int, u_soc, v_int, v_soc, y_do):
        
        
        z_full = torch.cat([u_int, u_soc, v_int, v_soc], dim=1)   
        y_full = self.f_R(z_full).squeeze(1)                        

        
        
        
        delta = y_full - y_do   

        return y_full, delta

    
    
    
    
    def heterogeneous_gate(self, u_int, u_soc, v_int, v_soc):
        
        
        z_int = torch.cat([u_int, v_int], dim=1)   

        
        gate_u_input = torch.cat([u_int, u_soc, z_int], dim=1)   
        alpha_u = torch.sigmoid(self.gate_u(gate_u_input))        

        
        gate_v_input = torch.cat([v_int, v_soc, z_int], dim=1)   
        alpha_v = torch.sigmoid(self.gate_v(gate_v_input))        

        
        
        h_u = u_int + alpha_u * u_soc   
        h_v = v_int + alpha_v * v_soc   

        return h_u, h_v, alpha_u.squeeze(1), alpha_v.squeeze(1)

    
    
    
    
    def alignment_loss(self, alpha_u, alpha_v, delta):
        
        
        gate_mean = (alpha_u + alpha_v) / 2.0   

        
        
        
        target = torch.sigmoid(delta).detach()   

        
        loss_align = F.mse_loss(gate_mean, target)

        return loss_align

    
    
    
    def forward(self, u_int, u_soc, v_int, v_soc, y_do, labels):
        

        
        
        
        y_full, delta = self.causal_gain_estimation(u_int, u_soc, v_int, v_soc, y_do)

        
        
        
        h_u, h_v, alpha_u, alpha_v = self.heterogeneous_gate(u_int, u_soc, v_int, v_soc)

        
        
        
        z_final = torch.cat([h_u, h_v, y_do.unsqueeze(1)], dim=1)   
        y_hat   = self.f_P(z_final).squeeze(1)                        

        
        loss_rec = F.mse_loss(y_hat, labels)

        
        loss_align = self.alignment_loss(alpha_u, alpha_v, delta)

        
        
        loss_dec = loss_rec + self.beta * loss_align

        return loss_dec, y_hat
