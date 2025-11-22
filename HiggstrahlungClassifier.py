import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lhereader import LHEReader
from itertools import islice
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def smear_vec(px,py,pz, rel=0.01):
    s = 1.0 + np.random.normal(0, rel)
    px2, py2, pz2 = s*px, s*py, s*pz
    E2 = (px2**2 + py2**2 + pz2**2)**0.5
    return px2, py2, pz2, E2

## Convert LHE file --> pd dataframe
def MakeDF_bbZ(file, smear): 
    pd_rows = []; 
    for iev, event in enumerate(LHEReader(file)):
        b1 = event.particles[-1] 
        b2 = event.particles[-2]
        Z = event.particles[-3]
        # Smear
        b1px, b1py, b1pz, b1E = smear_vec(b1.px,b1.py,b1.pz, rel=smear)
        b2px, b2py, b2pz, b2E = smear_vec(b2.px,b2.py,b2.pz, rel=smear)
    
        m_bb = np.sqrt(2*(b1E*b2E - b1px*b2px - b1py*b2py - b1pz*b2pz))

        pd_rows.append({
            "m_bb": m_bb, 
            "px_Z": Z.px,  "py_Z": Z.py,  "pz_Z": Z.pz,  "E_Z": Z.energy,
            "px_b1": b1.px, "py_b1": b1.py, "pz_b1": b1.pz, "E_b1": b1.energy,
            "px_b2": b2.px, "py_b2": b2.py, "pz_b2": b2.pz, "E_b2": b2.energy,
        })
    return pd.DataFrame(pd_rows)



class HiggsNN:
    def __init__(self, df_sig, df_bkg, epochs=25, batch_size=128, lr=1e-3, 
                 weight_decay=1e-4, layers_and_nodes=[32, 16]):
        X_sig = df_sig.values.astype(np.float32)
        X_bkg = df_bkg.values.astype(np.float32)
        X = np.vstack([X_sig, X_bkg])
        y = np.concatenate([np.ones(len(X_sig)), np.zeros(len(X_bkg))]).astype(np.float32)
        print(f"X shape: {X.shape},  y shape: {y.shape}")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Xtr shape: {Xtr.shape},  Xte shape: {Xte.shape}")
        print(f"ytr shape: {ytr.shape},  yte shape: {yte.shape}")

        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu)/sd;  Xte = (Xte - mu)/sd
        self.Xt, self.yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        self.Xv, self.yv = torch.from_numpy(Xte), torch.from_numpy(yte)

        input_dim = Xtr.shape[1]
       
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 1)
        )
        #~~~~
#         layers = []
#         prev = input_dim
#         for node in layers_and_nodes:
#             layers += [nn.Linear(prev, node), nn.ReLU()]
#             prev = node
#         layers += [nn.Linear(prev, 1)]
#         self.model = nn.Sequential(*layers)
        #~~~~
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_losses = []
        self.test_losses = []
        
        
    def evaluate_loss(self, X, y):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).squeeze(1)
            loss = self.loss_fn(preds, y)
        self.model.train()
        return loss.item()

    def fit(self):
        Xt, yt, n = self.Xt, self.yt, len(self.Xt)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                xb, yb = Xt[idx], yt[idx]
                self.opt.zero_grad()
                loss = self.loss_fn(self.model(xb).squeeze(1), yb)
                loss.backward(); 
                self.opt.step()
                epoch_loss += loss.item() * len(xb)
            # Training Loss
            train_loss = epoch_loss / n 
            self.train_losses.append(train_loss) 
            # Testing Loss
            test_loss  = self.evaluate_loss(self.Xv, self.yv)
            self.test_losses.append(test_loss)

    def plot_loss(self, train=False, test=False):
        plt.figure(figsize=(3, 2))
        if train: plt.plot(self.train_losses, lw=2)
        if test: plt.plot(self.test_losses, lw=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if train: plt.legend(['Training Loss'])
        if test: plt.legend(['Testing Loss'])
        if train and test:  plt.legend(['Training Loss', 'Testing Loss'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def predict_scores(self):
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(self.Xv).squeeze(1)).numpy()

    def roc_auc(self):
        with torch.no_grad():
            scores = torch.sigmoid(self.model(self.Xv).squeeze(1)).numpy()
        return roc_auc_score(self.yv, scores)
    
    
    
class HiggsAE(nn.Module):
    def __init__(self, df_sig, df_bkg, epochs=25,
                 batch_size=128, lr=1e-3, weight_decay=1e-4, latent=2,
                 layers_and_nodes=[32, 32, 8, 8]):
        super().__init__()
        X_sig = df_sig.values.astype(np.float32)
        X_bkg = df_bkg.values.astype(np.float32)
        
        Xb_tr, Xb_te = train_test_split(X_bkg, test_size=0.2, random_state=42)
        mu, sd = X_bkg.mean(0), X_bkg.std(0) + 1e-8
        Xb_tr = (Xb_tr - mu)/sd
        Xb_te = (Xb_te - mu)/sd
        Xs = (X_sig - mu)/sd
        
        self.Xb_tr_t = torch.from_numpy(Xb_tr)
        self.Xb_te_t = torch.from_numpy(Xb_te)
        self.Xs_t   = torch.from_numpy(Xs)
        
        ### ~~~~~~~~~~~~~~~~~~ ###
        ### Define Autoencoder ###
        ### ~~~~~~~~~~~~~~~~~~ ###
        d_input = Xb_tr.shape[1] 
        
        # ----- Encoder -----
        enc_layers = []
        prev = d_input
        for node in layers_and_nodes:
            enc_layers += [nn.Linear(prev, node), nn.ReLU()]
            prev = node
        enc_layers += [nn.Linear(prev, latent)]
        self.encoder = nn.Sequential(*enc_layers)

        # ----- Decoder -----
        dec_layers = []
        prev = latent
        for node in reversed(layers_and_nodes):
            dec_layers += [nn.Linear(prev, node), nn.ReLU()]
            prev = node
        dec_layers += [nn.Linear(prev, d_input)]
        self.decoder = nn.Sequential(*dec_layers)
        
        # ----- Autoencoder -----
        self.ae = nn.Sequential(self.encoder, self.decoder)
        
        
        self.opt = torch.optim.AdamW(self.ae.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs
        self.batch_size = batch_size 
        self.train_losses = []
        self.test_losses = []
        
    ## Count number of model parameters ... 
    def nparams(self): 
        return sum(p.numel() for p in self.ae.parameters())
    
    def evaluate_loss(self, X):
        """Compute mean reconstruction loss on any tensor X."""
        self.ae.eval()
        with torch.no_grad():
            recon = self.ae(X)
            loss = self.loss_fn(recon, X)
        self.ae.train()
        return loss.item()   # already averaged if loss_fn uses reduction='mean'
 
    ### ~~~~~~~~~~~~~ ###
    ### Train AE Model ###
    ### ~~~~~~~~~~~~~ ###
    def fit(self):
        n = len(self.Xb_tr_t)
        for epoch in range(self.epochs):
            self.ae.train()
            perm = torch.randperm(n)
            epoch_loss = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                xb = self.Xb_tr_t[idx]
                self.opt.zero_grad()
                recon = self.ae(xb)
                loss = self.loss_fn(recon, xb)
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item() * len(xb)
            # Training Loss
            train_loss = epoch_loss / n 
            self.train_losses.append(train_loss) 
            # Testing Loss
            test_loss  = self.evaluate_loss(self.Xb_te_t)
            self.test_losses.append(test_loss)

    def plot_loss(self, train=False, test=False):
        plt.figure(figsize=(3, 2))
        if train: plt.plot(self.train_losses, lw=2)
        if test: plt.plot(self.test_losses, lw=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if train: plt.legend(['Training Loss'])
        if test: plt.legend(['Testing Loss'])
        if train and test:  plt.legend(['Training Loss', 'Testing Loss'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
                
    def predict_scores(self):
        self.ae.eval()
        with torch.no_grad():
            rb = self.ae(self.Xb_te_t)              # recon of held-out background
            rs = self.ae(self.Xs_t)                 # recon of signal
        b_scores = ((self.Xb_te_t - rb)**2).mean(1).numpy()
        s_scores = ((self.Xs_t    - rs)**2).mean(1).numpy()
        return b_scores, s_scores
#             y = np.concatenate([np.zeros_like(b_scores), np.ones_like(s_scores)])
#             scores = np.concatenate([b_scores, s_scores])
        
    

    
    
    