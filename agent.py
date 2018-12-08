import numpy as np

import torch
import torch.nn as nn

BATCH_SIZE = 32    # minibatch size
GAMMA = 0.99       # discount rate
TAU = 0.95         #
LR = 3e-4          # learning rate

GRADIENT_CLIP = 5
ROLLOUT_LENGTH = 2048
NUM_EPOCHS = 10     # optimization epochs
PPO_CLIP = 0.2
LOG_INTERVAL = 2048

BETA = 0.01         # entropy coefficient 
LR = 3e-4           # Adam learning rate
EPSILON = 1e-5      # Adam epsilon

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]

class PPOAgent(object):
    
    def __init__(self, environment, brain_name, policy_network, optimizier, num_agents):
        """ Initialize an Agent object 
        Params
        ======
        
        """
        self.network = policy_network
        self.optimizier = optimizier
        self.total_steps = 0
        self.all_rewards = np.zeros(num_agents)
        self.episode_rewards = []
        self.environment = environment
        self.brain_name = brain_name
        self.num_agents = num_agents
        
        env_info = environment.reset(train_mode=True)[brain_name]    
        self.states = env_info.vector_observations              

    def step(self, num_agents):
        num_agents = self.num_agents
        rollout = []

        env_info = self.environment.reset(train_mode=True)[self.brain_name]    
        self.states = env_info.vector_observations  
        states = self.states
        for _ in range(ROLLOUT_LENGTH):
            actions, log_probs, _, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            terminals = np.array([1 if t else 0 for t in env_info.local_done])
            self.all_rewards += rewards
            
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.all_rewards[i])
                    self.all_rewards[i] = 0
                    
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.network(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.Tensor(np.zeros((num_agents, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = torch.Tensor(terminals).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            
            returns = rewards + GAMMA * terminals * returns

            td_error = rewards + GAMMA * terminals * next_value.detach() - value.detach()
            advantages = advantages * TAU * GAMMA * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        batcher = Batcher(states.size(0) // BATCH_SIZE, [np.arange(states.size(0))])
        for _ in range(NUM_EPOCHS):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = torch.Tensor(batch_indices).long()
                
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, entropy_loss, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * sampled_advantages
                
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - BETA * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizier.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), GRADIENT_CLIP)
                self.optimizier.step()

        steps = ROLLOUT_LENGTH * num_agents
        self.total_steps += steps
