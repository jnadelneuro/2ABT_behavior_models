#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:21:06 2021

@author: celiaberon
"""

import numpy as np
from scipy.optimize import minimize


def fit_with_scipy(ll_func, training_data, init_parameters=(1.0, 1.0, 1.0)):
    '''
    fit behavior model with scipy optimizer
    
    INPUTS:
        - ll_func (function): log likelihood function for given model
        - training_data (nested lists): [choices, rewards] by session 
        - init_parameters (tuple): starting parameters, varies in length by model
        
    OUTPUTS:
        - (np array) optimized parameters
        - nll: negative log likelihood
    '''
    
    loss_fn = lambda parameters: -ll_func(parameters, training_data)
    
    result = minimize(loss_fn, init_parameters, method='L-BFGS-B')
    
    return np.asarray(result.x), result.fun


'''RECURSIVELY FORMULATED LOGISTIC REGRESSION'''

def log_sum_exp_stable(x):
    """
    Numerically stable version of np.log(1 + np.exp(x))
    """
    if x > 30:
        # For large x, np.exp(x) dominates, so log(1 + exp(x)) is approx. log(exp(x)) = x
        return x
    elif x < -30:
        # For large negative x, np.exp(x) is close to 0, so log(1 + exp(x)) is approx. log(1) = 0
        return 0.0
    else:
        # The original calculation is safe for values in this range
        return np.log(1 + np.exp(x))

def _log_prob_single_rflr(parameters, choices, rewards):
    
    alpha, beta, tau = parameters  # unpack parameters
    gamma = np.exp(-1 / tau)
    
    ll = 0.0
    phi = beta * rewards[0] * (2 * choices[0] - 1)
    
    # Loop through trials
    for i in range(len(choices) - 1):
        prev_choice = choices[i]
        choice = choices[i + 1]
        reward = rewards[i + 1]
        
        # update
        psi = phi + alpha * (2 * prev_choice - 1)
        ll += choice * psi - log_sum_exp_stable(psi)
        phi = gamma * phi + beta * reward * (2 * choice - 1)
    
    return ll

    
def log_probability_rflr(parameters, sessions):
    
    # compute probability of next choice
    ll = 0.0
    n = 0
    for choices, rewards in sessions:        
        # initialize "belief state" for this session
        
        ll += _log_prob_single_rflr(parameters, choices, rewards)
        n += len(choices) - 1
            
    return ll / n
