import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam,RMSprop
from modules.utils import merge_dict, multinomials_log_density
import time
import random
from runner import Runner

import argparse

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))


class RunnerHiercommStructure(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)


        self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr = self.args.lr, alpha=0.97, eps=1e-6)

        self.n_nodes = int(self.n_agents * (self.n_agents - 1) / 2)
        self.interval = self.args.interval




    def random_god_action(self):
        return [np.array(random.randint(1, 10)).reshape(1)]


    def optimizer_zero_grad(self):
        self.optimizer_agent_ac.zero_grad()


    def optimizer_step(self):
        self.optimizer_agent_ac.step()



    def collect_batch_data(self, batch_size):
        batch_data = []
        batch_log = dict()
        num_episodes = 0

        while len(batch_data) < batch_size:
            episode_data, episode_log = self.run_an_episode()
            batch_data += episode_data
            merge_dict(episode_log, batch_log)
            num_episodes += 1

        batch_log['num_episodes'] = num_episodes
        batch_log['num_steps'] = len(batch_data)
        batch_data = Transition(*zip(*batch_data))

        return batch_data, batch_log




    def train_batch(self, batch_size):
        batch_data, batch_log = self.collect_batch_data(batch_size)
        self.optimizer_zero_grad()
        train_log = self.compute_grad(batch_data)
        merge_dict(batch_log, train_log)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        self.optimizer_step()
        return train_log




    def run_an_episode(self):

        log = dict()

        memory = []

        self.reset()
        obs = self.env.get_obs()

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

        cmatrix = self.agent.agent_clustering(obs_tensor)
        sets = self.agent.cmatrix_to_set(cmatrix)

        god_reward_list = []
        god_reward = np.zeros(1)

        step = 1
        num_group = 0
        episode_return = 0
        done = False

        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                cmatrix = self.agent.agent_clustering(obs_tensor)
                sets = self.agent.cmatrix_to_set(cmatrix)

            after_comm = self.agent.communicate(obs_tensor, sets)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)

            god_reward_list.append(np.mean(rewards).reshape(1))

            if step % self.interval == 0:
                god_reward = np.mean(god_reward_list).reshape(1)
                god_reward_list = []

            next_obs = self.env.get_obs()

            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            god_episode_mask = np.ones(1)
            if done:
                episode_mask = np.zeros(rewards.shape)
                god_episode_mask = np.zeros(1)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)

            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)

            # if step % self.interval == 0:
            #     god_trans = God_Transition(god_action_out, god_action, god_reward, god_value, god_episode_mask)
            #     god_memory.append(god_trans)

            obs = next_obs
            episode_return += int(np.sum(rewards))
            step += 1
            num_group += len(sets)

        log['episode_return'] = episode_return
        log['episode_steps'] = [step - 1]
        log['num_groups'] = num_group / (step - 1)

        if 'num_collisions' in env_info:
            log['num_collisions'] = int(env_info['num_collisions'])

        # if self.args.env == 'tj':
        #     merge_dict(self.env.get_stat(),log)

        return memory, log

    def compute_god_grad(self, batch):

        log = dict()
        batch_size = len(batch.god_value)
        n = 1

        rewards = torch.Tensor(np.array(batch.god_reward))
        actions = torch.Tensor(np.array(batch.god_action))
        actions = actions.transpose(1, 2).view(-1, n, 1)

        episode_masks = torch.Tensor(np.array(batch.episode_masks))

        values = torch.cat(batch.god_value, dim=0)
        action_outs = torch.stack(batch.god_action_out, dim=0)

        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)
        prev_returns = 0

        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + self.args.gamma * prev_returns * episode_masks[i]
            prev_returns = returns[i].clone()

        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        log_p_a = [action_outs.view(-1, 10)]
        actions = actions.contiguous().view(-1, 1)
        log_prob = multinomials_log_density(actions, log_p_a)
        action_loss = -advantages.view(-1) * log_prob.squeeze()
        actor_loss = action_loss.sum()

        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        critic_loss = value_loss.sum()

        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()

        log['god_action_loss'] = actor_loss.item()
        log['god_value_loss'] = critic_loss.item()
        log['god_total_loss'] = total_loss.item()



        return log




