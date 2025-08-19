from math import sqrt, log, pi

def expected_max_sharpe(n):
 if n<=1: return 0.0
 return sqrt(2*log(n)) - (log(log(n))+log(4*pi))/(2*sqrt(2*log(n)))

def deflated_sharpe(sr, n_obs, n_trials):
 z = sr*sqrt(max(n_obs-1,1))
 emax = expected_max_sharpe(max(n_trials,1))
 return (z/sqrt(max(n_obs,1))) - emax
