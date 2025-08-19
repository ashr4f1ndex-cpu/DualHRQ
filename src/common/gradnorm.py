import torch
class GradNorm:
 def __init__(self, task_weights, alpha=0.5):
  self.task_weights=task_weights; self.alpha=alpha; self.initial_losses=None
 def step(self, losses, shared_params):
  if self.initial_losses is None:
   with torch.no_grad(): self.initial_losses=torch.tensor([l.item() for l in losses], device=self.task_weights.device)
  G=[]
  for L in losses:
   grads=torch.autograd.grad(L, shared_params, retain_graph=True, allow_unused=True)
   g_norm=torch.sqrt(sum([(g**2).sum() for g in grads if g is not None])+1e-12); G.append(g_norm)
  G=torch.stack(G)
  with torch.no_grad():
   loss_ratios=torch.tensor([l.item() for l in losses], device=self.task_weights.device)/(self.initial_losses+1e-8)
   avg_G=G.mean(); target_G=avg_G*(loss_ratios**self.alpha)
  return (torch.abs(G-target_G.detach())*self.task_weights).sum()
