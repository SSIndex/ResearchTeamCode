'''
File with further scoe functions not available in situ
'''
import numpy as np
from sklearn.metrics import (
  log_loss
)

def aic(pred, truth, num_regressors=15):
  '''
  Akaike information criterion.  `-2*(llf - p)` where `p` is the number
  of regressors including the intercept.
  '''
  return -2 * (log_likelihood(pred, truth) - num_regressors)


def bic(pred, truth, num_observations=280, num_regressors=15):
  '''
  Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
  number of regressors including the intercept.
  implementation founded in: 
  https://github.com/statsmodels/statsmodels/blob/main/statsmodels/discrete/discrete_model.py#L4599
  '''
  return -2 * (log_likelihood(pred, truth)) + np.log(num_observations) * num_regressors


def log_likelihood(pred, truth):
  '''
  Log Likelihood score
  '''
  return -1 * log_loss(truth, pred) * len(truth)