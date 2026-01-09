#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:44:54 2021

@author: celiaberon
"""

import numpy as np
from functools import partial
from sklearn.linear_model import LogisticRegression
from scipy.special import expit as sigmoid
from scipy.stats import bernoulli
from tqdm.auto import trange
from scipy.special import logit


'''GENERAL'''

def model_to_policy(model_probs, sessions_data, policy='stochastic', **kwargs):
    
    '''relies on predefined histories and sorting
    INPUTS:
        model_probs (nested_list): posterior estimates from models for choice as probabilities
        sessions_data: usually for test dataset
        policy (str): decision policy off of model estimate; can be 'stochastic','greedy', or 'softmax'
        
    OUTPUTS:
        model_choice (nested list): predicted choices after applying policy
        model_switch: predicted switches, as trial-to-trial differences in predicted next choice from mouse previous choice
    '''
    
    def stochastic_pol(x): return int(np.random.rand() < x)
    
    def greedy_pol(x): return int(round(x))
    
    def softmax_pol(x, T): 
        choice_prob = sigmoid( logit(x) / T)
        return int(np.random.rand() < choice_prob) # same as stochastic_pol where probs have been filtered through softmax
    
    def make_choice(model_probs, mouse_choices, policy, **kwargs):
        
        '''
        use specified policy to make choice from each model estimate
        INPUTS:
            model_probs (nested list)
            mouse_choices (nested list)
            policy (str)
        OUTPUTS:
            predicted_choices (nested list)
            predicted_switches (nested list)
        '''
        
        predicted_choices, predicted_switches=[], []
        for session_probs, session_choice_history in zip(model_probs, mouse_choices):
            session_choices = [policy(model_prob[1]) for model_prob in session_probs]
            predicted_choices.append(session_choices)

            session_switches = [int(predicted_choice!=last_choice) for predicted_choice, last_choice \
                                in zip(session_choices[1:], session_choice_history[:-1])]
            session_switches.insert(0,0) # define first choice as not switch
            predicted_switches.append(session_switches)

        return predicted_choices, predicted_switches

    mouse_choices=[]
    for session_choices, session_rewards in sessions_data:
        mouse_choices.append(session_choices)
    
    if policy=='stochastic': model_choice, model_switch = make_choice(model_probs, mouse_choices, policy=stochastic_pol)
    elif policy=='greedy': model_choice, model_switch = make_choice(model_probs, mouse_choices, policy=greedy_pol)
    elif policy=='softmax': 
        T=kwargs.get('temp')
        model_choice, model_switch = make_choice(model_probs, mouse_choices, policy=partial(softmax_pol, T=T))
        
    return model_choice, model_switch


def log_likelihood_model_policy(policies, sessions):
       
    '''
    evaluate the per trial likelihood of each session
    INPUTS:
        - policies (nested lists): choice probabilities output by model
        - sessions (nested lists): [choice, reward] for each session (just need choice)
        
    OUTPUTS:
        - ll/n (float): average log-likelihood across all trials
    '''
    
    ll = 0
    n = 0
    
    for i in trange(len(sessions), disable=True):
        choices, rewards = sessions[i]
        policy = policies[i][:, 1] # P(choice==1)

        # Update the log likelihood estimate
        ll += bernoulli.logpmf(choices, policy).sum()
        n += len(choices) # number of trials
        
    return ll / n


def log_likelihood_empirical_policy(policy_df, test_sessions, memory):
    ll = 0
    n = 0

    for row in test_sessions.iterrows():
        pleft = policy_df[policy_df.history==row[1].RL_history].pdecision.item()
        pleft = np.clip(pleft, 1e-4, 1-(1e-4))
        ll += row[1].Decision * np.log(pleft) + (1 - row[1].Decision) * np.log(1 - pleft)
        n += 1
    return ll / n


'''LOGISTIC REGRESSION'''

pm1 = lambda x: 2 * x - 1
feature_functions = [
    lambda cs, rs: pm1(cs),                # choices
    lambda cs, rs: rs,                     # rewards
    lambda cs, rs: pm1(cs) * rs,           # +1 if choice = 1 and reward, 0 if no reward, -1 if choice=0 and reward
    lambda cs, rs: np.ones(len(cs))        # overall bias term
    
]

def encode_session(choices, rewards, memories, featfun):
    
    '''Helper to encode sessions in features and outcomes'''
    
    assert len(memories) == len(featfun)  
    
    # Construct the features
    features = []
    for fn, memory in zip(featfun, memories): 
        for lag in range(1, memory+1):
            # encode the data and pad with zeros
            x = fn(choices[:-lag], rewards[:-lag])
            x = np.concatenate((np.zeros(lag), x))
            features.append(x)
    features = np.column_stack(features)
    return features, choices
    
def fit_logreg_policy(sessions, memories, featfun=feature_functions):
    
    '''
    fit logistic regression to training data
    INPUTS:
        - sessions (nested lists): [choices, rewards] from each session of training dataset
        - memories (list): memory length for each feature from featfun
        -featfun: list of functions to encode input features for logistic regression
    OUTPUTS:
        -lr (LogisticRegression): fit model
    '''
    
    encoded_sessions = [encode_session(*session, memories, featfun=featfun) for session in sessions]
    X = np.row_stack([session[0] for session in encoded_sessions])
    y = np.concatenate([session[1] for session in encoded_sessions])
    
    # Construct the logistic regression model and fit to training sessions
    lr = LogisticRegression(C=1.0, fit_intercept=False)
    lr.fit(X, y)
    return lr

def compute_logreg_probs(sessions, lr_args, featfun=feature_functions):
    
    '''
    use fit logistic regression to calculate choice probabilities from mouse data
    INPUTS:
        - sessions (nested lists): [choices, rewards] from each session of mouse data 
        - lr_args: fit logistic regression and memory lengths for features
        - featfun: list of functions to encode input features for logistic regression
    OUTPUTS:
        - policies (nested lists): choice probabilities by trial for each session 
    '''
    lr, memories = lr_args
    
    policies = []
    for choices, rewards in sessions:
        X, y = encode_session(choices, rewards, memories, featfun=featfun)
        policy = lr.predict_proba(X)#[:, 1]
        policies.append(policy)
    return policies


'''RECURSIVELY FORMULTATED LOGISTIC REGRESSION (RFLR)'''
        
def RFLR(behavior_features, parameters):
    
    '''
    trial-by-trial calculatio ofn choice probabilities using a recursively formulated logistic regression;
    reinitializes every session, uses mouse behavior as model history
    INPUTS:
        - behavior_features (nested lists): [choices, rewards] from each session of mouse dataset
        - parameters (tuple): alpha, beta, tau as fit with sgd
    OUTPUTS:
        - psi_sessions (nested lists): choice probabilities by trial for each session
            
    '''

    alpha, beta, tau = parameters  # unpack parameters
    
    gamma = np.exp(-1 / tau)
    
    psi_sessions=[]

    for choices, rewards in behavior_features:
        
        # initialize psi
        psi=np.zeros((len(choices), 2))
        psi[0,:]=[0.5,0.5] 
    
        # recode choices
        cbar = 2 * choices - 1
        
        # initialize "belief state"
        phi = beta * rewards[0] * cbar[0]
        
        for t in range(1, len(choices)):
            
            # evaluate probability of this choice
            psi[t,:] = 1-sigmoid(phi + (alpha * cbar[t-1])), sigmoid(phi + (alpha * cbar[t-1]))
            
            # update belief state for next time step
            phi = gamma * phi + (beta*(rewards[t] * cbar[t])) 

        psi_sessions.append(psi)

    return psi_sessions


'''Q-LEARNING'''

def fq_learning_model(behavior_features, parameters):
    
    '''
    trial by trial calculation choice probabilities using a F-Q-learning algorithm
    reinitializes every session, uses mouse behavior as model history
    
    INPUTS:
        - df (nested lists): [choices, rewards] from each session of mouse dataset
        - parameters (tuple): alpha (choice history bias), k (learning=forgetting rate), T (softmax temperature) as derived from Logistic Regression
        
    OUTPUTS:
        - psi_sessions (nested lists): choice probabilities by trial for each session
    '''
    
    def update_q(q, choice, reward):
        
        '''update Q-values based on choice direction and choice outcome'''
        
        q[choice] = (k * (reward - q[choice])) + q[choice] # ((1-beta)*q[choice]) # 
        q[1-choice] = (1-k)*q[1-choice]
        
        return q
    
    alpha, k, T = parameters  # unpack parameters
        
    psi_sessions, q_sessions = [], [] 

    for choices, rewards in behavior_features:
        
        psi=np.zeros((len(choices), 2))
        psi[0,:]=[0.5,0.5]
        q = np.zeros_like(psi)
        q[0,:] = [0, 0]

        for t in range(1, len(choices)):
            
            # evaluate probability of this choice
            psi[t,1] = sigmoid( ( (q[t-1,1] - q[t-1,0])/T ) + (alpha * (2 * choices[t-1] - 1)) )
            psi[t,0] = 1 - psi[t,1]
            
            # update q for next trial
            q[t,:] = update_q(q[t-1,:], choices[t], rewards[t]) # q_t+1
               
        psi_sessions.append(psi)
        q_sessions.append(q)

    return psi_sessions
