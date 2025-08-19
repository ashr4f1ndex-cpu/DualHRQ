import numpy as np
from itertools import combinations

def cpcv_splits(n_samples, n_groups=6, test_groups=2, embargo=0):
 groups=np.array_split(np.arange(n_samples), n_groups); idx=np.arange(n_groups)
 for comb in combinations(idx, test_groups):
  test_idx=np.concatenate([groups[i] for i in comb])
  train_groups=[i for i in idx if i not in comb]
  train_idx=np.concatenate([groups[i] for i in train_groups])
  if embargo>0:
   tmin,tmax=test_idx.min(),test_idx.max(); mask=(train_idx<tmin-embargo)|(train_idx>tmax+embargo); train_idx=train_idx[mask]
  yield np.sort(train_idx), np.sort(test_idx)
