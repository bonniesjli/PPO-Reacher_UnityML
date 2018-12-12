import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import ActorCritic

device   = torch.device("cpu")

BATCH_SIZE = 32    # minibatch size
GAMMA = 0.99       # discount rate
TAU = 0.95         

GRADIENT_CLIP = 5
ROLLOUT_LENGTH = 2048
NUM_EPOCHS = 10     # optimization epochs
PPO_CLIP = 0.2

BETA = 0.01         # entropy coefficient 
LR = 3e-4           # Adam learning rate
EPSILON = 1e-5      # Adam epsilon

class Agent(object):
    """Interacts and learns from the environment"""
    
    def __init__(self, num_agents, state_size, action_size):
        """ Initialize an Agent object 
        
        Params
        ======
            num_agent (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.num_agents = num_agents
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = ActorCritic(state_size, action_size, 256)
        self.optimizer = optim.Adam(self.model.parameters(), LR, eps = EPSILON)
        

    def step(self, rollout, returns, num_agents):
        """ Given trajectory, compute advantage estimates at each time steps"""
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.Tensor(np.zeros((num_agents, 1)))

        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, dones = rollout[i]
            dones = torch.Tensor(dones).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            
            # V(s) = r + γ * V(s')
            returns = rewards + GAMMA * dones * returns
            
            # L = r + γ*V(s') - V(s)
            td_error = rewards + GAMMA * dones * next_value.detach() - value.detach()
            
            advantages = advantages * TAU * GAMMA * dones + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        self.learn(states, actions, log_probs_old, returns, advantages, num_agents)
   
    def act(self, states):
        """Given state as per current policy model, returns action, log probabilities and estimated state values"""
        dist, values = self.model(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)

        return actions, log_probs, values
    
    def batcher(self, BATCH_SIZE, states, actions, log_probs_old, returns, advantages):
        """Convert trajectories into learning batches."""
        # for _ in range(states.size(0) // BATCH_SIZE):
        rand_ids = np.random.randint(0, states.size(0), BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs_old[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]
        
    def learn(self, states, actions, log_probs_old, returns, advantages, num_agents):
        """ Optimize surrogate loss with policy and value parameters using given learning batches.
        
        1. Construct surrogate loss on NT timesteps of data
            L_CLIP(θ) = E { min[ r(θ)A, clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * KL }
            L_VF(θ) = ( V^(s) - V(s) )^2
            
        2. Optimize it with minibatch SGD, with K Epochs and minibatch size M <= NT
        """

        for _ in range(NUM_EPOCHS):
            # for _ in range(states.size(0) // BATCH_SIZE):
                
            for sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages in self.batcher(BATCH_SIZE, states, actions, log_probs_old, returns, advantages):

                dist, values = self.model(sampled_states)
                
                log_probs = dist.log_prob(sampled_actions)
                log_probs = torch.sum(log_probs, dim=1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # r(θ) =  π(a|s) / π_old(a|s)
                ratio = (log_probs - sampled_log_probs_old).exp()
                
                # Surrogate Objctive : L_CPI(θ) = r(θ) * A
                obj = ratio * sampled_advantages
                
                # clip ( r(θ), 1-Ɛ, 1+Ɛ )*A
                obj_clipped = ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * sampled_advantages
                
                # L_CLIP(θ) = E { min[ r(θ)A, clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * KL }
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - BETA * entropy.mean()
                
                # L_VF(θ) = ( V(s) - V_t )^2
                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()
               

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                self.optimizer.step()