import numpy as np
from scipy.stats import norm

def iv_term_structure(iv_by_ttm):
 items=sorted(iv_by_ttm.items()); ttms=np.array([k for k,_ in items],float); ivs=np.array([v for _,v in items],float)
 slope=(ivs[-1]-ivs[0])/(ttms[-1]-ttms[0]+1e-8) if len(ivs)>1 else 0.0
 curv=np.polyfit(ttms, ivs, 2)[0] if len(ivs)>=3 else 0.0
 return {'iv_ts_slope':slope,'iv_ts_curv':float(curv)}

def iv_skew(ivs_by_delta):
 d=ivs_by_delta; rr=(d.get(0.25,np.nan)-d.get(0.75,np.nan)); fly=d.get(0.25,np.nan)-2*d.get(0.5,np.nan)+d.get(0.75,np.nan)
 return {'rr_25':float(rr),'bf_25':float(fly)}

def smile_metrics(ivs_by_strike):
 ks=np.array(sorted(ivs_by_strike.keys()),float); ivs=np.array([ivs_by_strike[k] for k in ks],float)
 a=np.polyfit(ks,ivs,2)[0] if len(ks)>=3 else 0.0
 return {'smile_curv':float(a)}
