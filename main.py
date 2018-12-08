# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:33:12 2018

@author: lsj
"""

import sys
import random
from collections import namedtuple, deque

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from unityagents import UnityEnvironment
from IPython.display import clear_output

from model import PPOPolicyNetwork
from agent import PPOAgent

import tqdm

env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

config = {
    'environment': {
        'state_size':  env_info.vector_observations.shape[1],
        'action_size': brain.vector_action_space_size,
        'number_of_agents': len(env_info.agents)
    },
    'pytorch': {
        'device': torch.device("cpu")
    },
    'hyperparameters': {
        'discount_rate': 0.99,
        'tau': 0.95,
        'gradient_clip': 5,
        'rollout_length': 2048,
        'optimization_epochs': 10,
        'ppo_clip': 0.2,
        'log_interval': 2048,
        'max_steps': 1e5,
        'mini_batch_number': 32,
        'entropy_coefficent': 0.01,
        'episode_count': 250,
        'hidden_size': 512,
        'adam_learning_rate': 3e-4,
        'adam_epsilon': 1e-5
    }
}
    
def play_round(env, brain_name, policy, config):
    env_info = env.reset(train_mode=True)[brain_name]    
    states = env_info.vector_observations                 
    scores = np.zeros(config['environment']['number_of_agents'])                         
    while True:
        actions, _, _, _ = policy(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)
    
def ppo(env, brain_name, policy, config, train):
    if train:
        optimizier = optim.Adam(policy.parameters(), config['hyperparameters']['adam_learning_rate'], 
                        eps=config['hyperparameters']['adam_epsilon'])
        agent = PPOAgent(env, brain_name, policy, optimizier, config)
        all_scores = []
        averages = []
        last_max = 30.0
        
        for i in tqdm.tqdm(range(config['hyperparameters']['episode_count'])):
            agent.step()
            last_mean_reward = play_round(env, brain_name, policy, config)
            last_average = np.mean(np.array(all_scores[-100:])) if len(all_scores) > 100 else np.mean(np.array(all_scores))
            all_scores.append(last_mean_reward)
            averages.append(last_average)
            if last_average > last_max:
                torch.save(policy.state_dict(), f"models/ppo-max-hiddensize-{config['hyperparameters']['hidden_size']}.pth")
                last_max = last_average
            clear_output(True)
            print('Episode: {} Total score this episode: {} Last {} average: {}'.format(i + 1, last_mean_reward, min(i + 1, 100), last_average))
        return all_scores, averages
    else:
        score = play_round(env, brain_name, policy, config)
        return [score], [score]
    
new_policy = PPOPolicyNetwork(config)
all_scores, average_scores = ppo(env, brain_name, new_policy, config, train=True)