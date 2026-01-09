import os
import numpy as np
import pandas as pd
import seaborn as sns
import pyarrow.feather as feather
import matplotlib.pyplot as plt

import plot_models_v_mouse as bp
import model_policies as models
from sklearn.model_selection import train_test_split
import conditional_probs as cprobs
import resample_and_model_reps as reps
import model_fitting as fit

#%%
os.chdir(r'C:\Users\jan7154\Documents\GitHub\2ABT_behavior_models')

def calculate_block_length(row):
    """Calculates the TOTAL length of the block each trial belongs to."""
    block_type = np.array(row['blockTypeArray'])
    if block_type.size == 0:
        return []

    # Find the indices where a block change occurs
    change_indices = np.where(block_type[1:] != block_type[:-1])[0]
    
    # Get the start index of each block (0, plus where changes happen)
    block_starts = np.insert(change_indices + 1, 0, 0)
    
    # Get the length of each block by finding the difference between start indices
    block_lengths = np.diff(np.append(block_starts, block_type.size))
    
    # Create the output array
    block_length_array = np.zeros_like(block_type)
    
    # Fill the array with the corresponding block length for each trial
    for i, start_index in enumerate(block_starts):
        length = block_lengths[i]
        block_length_array[start_index : start_index + length] = length
        
    return block_length_array.tolist()


def calculate_target(row):
    """Calculates the 'Target' array for a session."""
    choices = np.array(row['choiceArray'])
    rewards = np.array(row['rewardArray'])
    if choices.size == 0:
        return []
    targets = np.where(rewards == 1, choices, 1 - choices)
    return targets.tolist()

# ---- THIS FUNCTION IS CORRECTED AND MORE ROBUST ----
def calculate_block_trial(row):
    """Calculates the trial number within each block."""
    block_type = row['blockTypeArray']
    if not isinstance(block_type, (list, np.ndarray)) or len(block_type) == 0:
        return []
    
    block_trials = []
    # If the list is not empty, the first trial is always 1
    block_trials.append(1) 
    
    # Iterate from the second trial to the end
    for i in range(1, len(block_type)):
        # If the block type is the same as the previous trial, increment counter
        if block_type[i] == block_type[i-1]:
            block_trials.append(block_trials[i-1] + 1)
        # Otherwise, reset the counter to 1 for the new block
        else:
            block_trials.append(1)
            
    return block_trials

def calculate_switch(row):
    """Calculates the 'Switch' array for a session."""
    choice_array = np.array(row['choiceArray'])
    if choice_array.size == 0:
        return []
    
    switches = np.zeros_like(choice_array, dtype=float)
    switches[1:] = np.where(choice_array[1:] != choice_array[:-1], 1, 0)
    return switches.tolist()

#%%
behaviorData = pd.read_feather('dataChoiceRew')
behaviorData = behaviorData.drop(columns=['win_switch_trials', 'lose_stay_trials', 'win_stay_trials', 'lose_switch_trials', 'total_wins', 'total_losses', 'sensor', 'siglocs', 'expgrp', 'implant'])
behaviorData = behaviorData[(behaviorData['total_trials'] <= 1000) & (behaviorData['sesType'] != 'FR1')] 
behaviorData['date'] = behaviorData['date'].astype('float').astype('int').astype('str')
behaviorData['Session'] = behaviorData['mouse'] + '_' + behaviorData['date']
behaviorData = behaviorData.drop(columns=['date', 'total_trials'])
behaviorData = behaviorData.rename(columns={'sesType' : 'Condition'})

behaviorData['targetArray'] = behaviorData.apply(calculate_target, axis=1)
behaviorData['blockTrialArray'] = behaviorData.apply(calculate_block_trial, axis=1)
behaviorData['switchArray'] = behaviorData.apply(calculate_switch, axis=1)
behaviorData['blockLengthArray'] = behaviorData.apply(calculate_block_length, axis=1) # Added this line

behaviorData = behaviorData.rename(columns={'choiceArray': 'Decision', 'rewardArray': 'Reward'})

# 3. Define the complete list of columns to explode
cols_to_explode = [
    'Decision', 
    'Reward', 
    'targetArray', 
    'blockTrialArray', 
    'switchArray',
    'blockLengthArray' # Added this line
]


# 3. Explode the DataFrame to create one row per trial
data_long = behaviorData.explode(cols_to_explode).reset_index(drop=True)

# 4. Rename the newly created columns to their final names
data_long = data_long.rename(columns={
    'targetArray': 'Target',
    'blockTrialArray': 'blockTrial',
    'switchArray': 'Switch',
    'blockLengthArray': 'blockLength' # Added this line
})

asyn_behavior = data_long[data_long['group'] == 'asyn']
control_behavior = data_long[data_long['group'] == 'control']

#%%

prob_list = ['100_0', '90_10', '75_25'] # List of probability conditions
seq_nback=2 # history length for conditional probabilites
train_prop=0.8 # for splitting sessions into train and test
# seed = np.random.randint(1000) # set seed for reproducibility

# Define groups to iterate over
groups = {'asyn': asyn_behavior, 'control': control_behavior}
results = []
individual_mouse_results = []  # Store per-mouse parameters for between-group comparisons

os.chdir(r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Bita\ASAP - Jillian Paper\official analysis\JIMMY\behavior\output_datafiles\RLFR modeling')

for probs in prob_list:
    for group_name, group_data in groups.items():
        print(f"Processing group: {group_name}, Condition: {probs}")
        
        data = group_data.loc[group_data.Condition==probs].copy() # segment out task condition
        
        if data.empty:
            print(f"No data for {group_name} in {probs} condition.")
            continue

        data = cprobs.add_history_cols(data, seq_nback) # set history labels up front

        train_session_ids, test_session_ids = train_test_split(data.Session.unique(), 
                                                               train_size=train_prop) # split full df for train/test

        data['block_pos_rev'] = data['blockTrial'] - data['blockLength'] # reverse block position from transition
        data['model']='mouse'
        data['highPort'] = data['Decision']==data['Target'] # boolean, chose higher probability port

        train_features, _, _ = reps.pull_sample_dataset(train_session_ids, data)
        test_features, _, block_pos_core = reps.pull_sample_dataset(test_session_ids, data)

        bpos_mouse = bp.get_block_position_summaries(block_pos_core)
        bpos_mouse['condition'] = 'mouse'

        # full dataset for sorting
        df_mouse_symm_reference = cprobs.calc_conditional_probs(data, symm=True, 
                                                                action=['Switch']).sort_values('pswitch')
        df_mouse_symm = cprobs.calc_conditional_probs(block_pos_core, symm=True, action=['Switch', 'Decision'])
        df_mouse_symm = cprobs.sort_cprobs(df_mouse_symm, df_mouse_symm_reference.history.values)
        df_mouse_symm.to_csv(f'{group_name}_{probs}_mouse_conditional_probs.csv', index=False)
        
        # Plot and save sequences (mouse only)
        plt.figure()
        bp.plot_sequences(df_mouse_symm, alpha=0.5) 
        plt.title(f'{group_name} {probs} Mouse Sequences')
        plt.savefig(f'{group_name}_{probs}_mouse_sequences.png')
        plt.close()

        L1 = 1 # choice history
        L2 =  5 # choice * reward history
        L3 = 0
        memories = [L1, L3, L2, 1]

        
        lr = models.fit_logreg_policy(train_features, memories) # refit model with reduced histories, training set
        model_probs = models.compute_logreg_probs(test_features, lr_args=[lr, memories])

        print(f'starting model fitting... for {group_name} {probs}')
        params, nll = fit.fit_with_scipy(fit.log_probability_rflr, train_features) # quick fit on RFLR parameters
        alpha, beta, tau = params
        print(f'alpha = {alpha:.2f}')
        print(f'beta = {beta:.2f}')
        print(f'tau = {tau:.2f}')
        
        # Store parameters
        results.append({
            'group': group_name,
            'condition': probs,
            'alpha': alpha,
            'beta': beta,
            'tau': tau,
            'nll': nll
        })

        # Fit individual mouse parameters for between-group comparisons
        print(f'Fitting individual mouse parameters for {group_name} {probs}...')
        mice_in_group = data['mouse'].unique()
        for mouse_id in mice_in_group:
            mouse_data = data[data['mouse'] == mouse_id]
            mouse_sessions = mouse_data['Session'].unique()
            
            if len(mouse_sessions) < 1:
                print(f'  Skipping mouse {mouse_id}: no sessions')
                continue
            
            mouse_features, _, _ = reps.pull_sample_dataset(mouse_sessions, mouse_data)
            
            try:
                mouse_params, mouse_nll = fit.fit_with_scipy(fit.log_probability_rflr, mouse_features)
                mouse_alpha, mouse_beta, mouse_tau = mouse_params
                
                individual_mouse_results.append({
                    'mouse': mouse_id,
                    'group': group_name,
                    'condition': probs,
                    'alpha': mouse_alpha,
                    'beta': mouse_beta,
                    'tau': mouse_tau,
                    'nll': mouse_nll,
                    'n_sessions': len(mouse_sessions)
                })
                print(f'  Mouse {mouse_id}: alpha={mouse_alpha:.2f}, beta={mouse_beta:.2f}, tau={mouse_tau:.2f}')
            except Exception as e:
                print(f'  Error fitting mouse {mouse_id}: {e}')
                continue

        model_probs = models.RFLR(test_features, params)

        T = (1-np.exp(-1/tau))/beta 
        k = 1-np.exp(-1/tau) 
        a = alpha 

        model_probs_QL = models.fq_learning_model(test_features, parameters=[a, k, T])

        model_choices, model_switches = models.model_to_policy(model_probs, test_features, policy='stochastic')

        block_pos_model = reps.reconstruct_block_pos(block_pos_core, model_choices, model_switches)
        bpos_model = bp.get_block_position_summaries(block_pos_model)
        bpos_model['condition'] = 'model' # label model predictions as such
        bpos_model_v_mouse = pd.concat((bpos_mouse, bpos_model)) # agg df with model predictions and mouse data
        color_dict = {'mouse': 'gray', 'model': sns.color_palette()[0]}#plot_config['model_seq_col']}
        
        # Plot and save block positions
        plt.figure()
        bp.plot_by_block_position(bpos_model_v_mouse, subset='condition', color_dict = color_dict)
        plt.title(f'{group_name} {probs} Block Position')
        plt.savefig(f'{group_name}_{probs}_block_position.png')
        plt.close()

        symm_cprobs_model = cprobs.calc_conditional_probs(block_pos_model, symm=True, action=['Switch'])
        symm_cprobs_model = cprobs.sort_cprobs(symm_cprobs_model, df_mouse_symm.history.values)
        
        # Plot and save model overlay
        plt.figure()
        bp.plot_sequences(df_mouse_symm, overlay=symm_cprobs_model, main_label='mouse', overlay_label='model')
        plt.title(f'{group_name} {probs} conditional probs')
        plt.savefig(f'{group_name}_{probs}_conditional_probs.png')
        plt.close()

        # Plot and save scatter
        plt.figure()
        bp.plot_scatter(df_mouse_symm, symm_cprobs_model)
        plt.title(f'{group_name} {probs} Scatter')
        plt.savefig(f'{group_name}_{probs}_scatter.png')
        plt.close()

# Save parameters to CSV
pd.DataFrame(results).to_csv('model_parameters.csv', index=False)

# Save individual mouse parameters to CSV for between-group comparisons
individual_mouse_df = pd.DataFrame(individual_mouse_results)
individual_mouse_df.to_csv('individual_mouse_parameters.csv', index=False)
print(f'\nSaved {len(individual_mouse_results)} individual mouse parameter fits to individual_mouse_parameters.csv')