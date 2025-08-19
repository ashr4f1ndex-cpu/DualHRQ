import os,random,numpy as np
try:
 import torch
except Exception:
 torch=None

def set_global_determinism(seed=42):
 os.environ['PYTHONHASHSEED']=str(seed)
 random.seed(seed); np.random.seed(seed)
 if torch is not None:
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
