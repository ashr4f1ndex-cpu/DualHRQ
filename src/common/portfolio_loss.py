import torch
EPS=1e-8

def to_bt_returns(logits):
 return torch.tanh(logits)

def sharpe_loss(R):
 return -((R.mean(dim=-1))/(R.std(dim=-1)+EPS)).mean()

def sortino_loss(R):
 D=torch.clamp_min(-R,0.0); ddev=torch.sqrt((D**2).mean(dim=-1)+EPS); return -((R.mean(dim=-1))/(ddev+EPS)).mean()

def cvar_loss(R, alpha=0.05):
 q=torch.quantile(R, alpha, dim=-1, keepdim=True, interpolation='linear'); tail=R[R<=q]; return tail.mean() if tail.numel()>0 else R.mean()*0.0

def evar_loss(R, alpha=0.05):
 lam=torch.sqrt(torch.tensor(2.0))*torch.erfinv(torch.tensor(2*alpha-1)); return torch.logsumexp(-lam*R, dim=-1).mean()
