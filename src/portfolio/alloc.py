import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def inverse_volatility(cov):
 iv=1.0/np.sqrt(np.diag(cov)+1e-12); return iv/iv.sum()

def min_variance(cov):
 ones=np.ones(cov.shape[0]); inv=np.linalg.pinv(cov+1e-8*np.eye(cov.shape[0])); w=inv@ones; return w/w.sum()

def hrp(cov):
 corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov))+1e-12)
 dist=np.sqrt(0.5*(1-corr)); link=linkage(squareform(dist,checks=False),'single'); order=dendrogram(link,no_plot=True)['leaves']
 cov_=cov[np.ix_(order,order)]; w=np.ones(cov_.shape[0]); clusters=[np.arange(cov_.shape[0])]
 while any(len(c)>1 for c in clusters):
  new=[]
  for c in clusters:
   if len(c)<=1: new.append(c); continue
   s=len(c)//2; c1,c2=c[:s],c[s:]
   var1=(w[c1].T@cov_[np.ix_(c1,c1)]@w[c1]); var2=(w[c2].T@cov_[np.ix_(c2,c2)]@w[c2])
   a=1-var1/(var1+var2+1e-12); w[c1]*=a; w[c2]*=(1-a); new.extend([c1,c2])
  clusters=new
 out=np.zeros_like(w); out[order]=w/w.sum(); return out
