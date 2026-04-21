import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WindowLayer(nn.Module):
    def __init__(self, hidden_size, num_gaussians, char_dim):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.linear = nn.Linear(hidden_size, 3 * num_gaussians)
        # Initialize kappa_hat biases to -5.0 to prevent premature window jumping
        nn.init.constant_(self.linear.bias[2 * num_gaussians:], -5.0)

    def forward(self, h1, prev_kappa, text_encoded):
        '''
        h1: [batch_size, hidden_size]
        prev_kappa: [batch_size, num_gaussians]
        text_encoded: [batch_size, seq_len, char_dim]
        '''
        params = self.linear(h1)
        
        # log weights for numerical stability
        alpha_hat, beta_hat, kappa_hat = torch.chunk(params, 3, dim=-1)
        
        alpha = torch.exp(alpha_hat)
        beta = torch.exp(beta_hat)
        # clamp kappa_hat to avoid extreme jumps
        kappa = prev_kappa + torch.exp(torch.clamp(kappa_hat, -10, 5)) 
        
        B = h1.size(0)
        U = text_encoded.size(1)
        K = self.num_gaussians
        
        u = torch.arange(1, U + 1, dtype=torch.float32, device=h1.device)
        u = u.unsqueeze(0).unsqueeze(0).expand(B, K, U)
        
        kappa_exp = kappa.unsqueeze(2)
        beta_exp = beta.unsqueeze(2)
        alpha_exp = alpha.unsqueeze(2)
        
        phi = torch.sum(alpha_exp * torch.exp(-beta_exp * (kappa_exp - u)**2), dim=1) # [B, U]
        phi_exp = phi.unsqueeze(2)
        
        w = torch.sum(phi_exp * text_encoded, dim=1) # [B, char_dim]
        
        return w, kappa, phi

class MDNLayer(nn.Module):
    def __init__(self, input_size, num_mixtures):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.linear = nn.Linear(input_size, 1 + 6 * num_mixtures)

    def forward(self, x):
        res = self.linear(x)
        
        eos_hat = res[:, 0:1] 
        pi_hat = res[:, 1:1+self.num_mixtures]
        mu_x = res[:, 1+self.num_mixtures:1+2*self.num_mixtures]
        mu_y = res[:, 1+2*self.num_mixtures:1+3*self.num_mixtures]
        sigma_x_hat = res[:, 1+3*self.num_mixtures:1+4*self.num_mixtures]
        sigma_y_hat = res[:, 1+4*self.num_mixtures:1+5*self.num_mixtures]
        rho_hat = res[:, 1+5*self.num_mixtures:1+6*self.num_mixtures]
        
        pi = F.softmax(pi_hat, dim=-1)
        sigma_x = torch.exp(sigma_x_hat) + 1e-6
        sigma_y = torch.exp(sigma_y_hat) + 1e-6
        rho = torch.tanh(rho_hat) * 0.999 # clamp pro stabilitu GMM
        
        # Pro autocast (mixed precision) se sigmoid nesmí volat před BCE rolí, 
        # vracíme proto syrové eos_hat (logits).
        return eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho

class HandwritingSynthesisNetwork(nn.Module):
    """
    Graves 2013 architecture pro rtx 5060 s optimalizací.
    """
    def __init__(self, vocab_size, char_embedding_dim=64, lstm_hidden_size=400, window_gaussians=10, mdn_mixtures=20):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, char_embedding_dim, padding_idx=0)
        
        self.lstm1 = nn.LSTMCell(3 + char_embedding_dim, lstm_hidden_size)
        self.window = WindowLayer(lstm_hidden_size, window_gaussians, char_embedding_dim)
        self.lstm2 = nn.LSTMCell(3 + lstm_hidden_size + char_embedding_dim, lstm_hidden_size)
        self.lstm3 = nn.LSTMCell(3 + lstm_hidden_size + char_embedding_dim, lstm_hidden_size)
        self.mdn = MDNLayer(lstm_hidden_size * 3, mdn_mixtures)
        
        self.hidden_size = lstm_hidden_size
        self.window_gaussians = window_gaussians

    def forward(self, x, text_tokens):
        # x: [B, T, 3] (dx, dy, eos)
        # text_tokens: [B, U] rozměr sekvence
        
        B, T, _ = x.size()
        device = x.device
        
        text_encoded = self.char_embed(text_tokens) # [B, U, char_dim]
        
        h1 = torch.zeros(B, self.hidden_size, device=device)
        c1 = torch.zeros(B, self.hidden_size, device=device)
        h2 = torch.zeros(B, self.hidden_size, device=device)
        c2 = torch.zeros(B, self.hidden_size, device=device)
        h3 = torch.zeros(B, self.hidden_size, device=device)
        c3 = torch.zeros(B, self.hidden_size, device=device)
        
        w = torch.zeros(B, text_encoded.size(2), device=device)
        kappa = torch.zeros(B, self.window_gaussians, device=device)
        
        outputs = []
        
        for t in range(T):
            x_t = x[:, t, :]
            
            lstm1_in = torch.cat([x_t, w], dim=-1)
            h1, c1 = self.lstm1(lstm1_in, (h1, c1))
            
            w, kappa, _ = self.window(h1, kappa, text_encoded)
            
            lstm2_in = torch.cat([x_t, h1, w], dim=-1)
            h2, c2 = self.lstm2(lstm2_in, (h2, c2))
            
            lstm3_in = torch.cat([x_t, h2, w], dim=-1)
            h3, c3 = self.lstm3(lstm3_in, (h3, c3))
            
            mdn_in = torch.cat([h1, h2, h3], dim=-1)
            out_t = self.mdn(mdn_in)
            outputs.append(out_t)
            
        eos_hat_stacked = torch.stack([o[0] for o in outputs], dim=1)
        pi = torch.stack([o[1] for o in outputs], dim=1)
        mu_x = torch.stack([o[2] for o in outputs], dim=1)
        mu_y = torch.stack([o[3] for o in outputs], dim=1)
        sigma_x = torch.stack([o[4] for o in outputs], dim=1)
        sigma_y = torch.stack([o[5] for o in outputs], dim=1)
        rho = torch.stack([o[6] for o in outputs], dim=1)
        
        return eos_hat_stacked, pi, mu_x, mu_y, sigma_x, sigma_y, rho

def mdn_loss(eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho, target_x, target_y, target_eos, mask):
    z_x = (target_x.unsqueeze(2) - mu_x) / sigma_x
    z_y = (target_y.unsqueeze(2) - mu_y) / sigma_y
    z = z_x**2 + z_y**2 - 2 * rho * z_x * z_y
    
    norm = 1.0 / (2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2 + 1e-8))
    exp_term = torch.exp(-z / (2 * (1 - rho**2 + 1e-8)))
    
    gaussian_probs = norm * exp_term 
    gmm_prob = torch.sum(pi * gaussian_probs, dim=2) + 1e-6 
    loss_gmm = -torch.log(gmm_prob)
    
    # BCEWithLogits = numericky bezpečná varianta v autocastu
    loss_eos = F.binary_cross_entropy_with_logits(eos_hat.squeeze(-1), target_eos, reduction='none')
    total_loss = loss_gmm + loss_eos
    
    total_loss = total_loss * mask
    return torch.sum(total_loss) / (torch.sum(mask) + 1e-8)
