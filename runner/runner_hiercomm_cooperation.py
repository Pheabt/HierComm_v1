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

Transition      = namedtuple('Transition',      ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
Team_Transition = namedtuple('Team_Transition', ('team_action_outs', 'team_actions', 'global_reward', 'global_value', 'episode_masks',))

class RunnerHiercommCooperation(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)


        self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr = self.args.lr, alpha=0.97, eps=1e-6)
        self.optimizer_team_ac = RMSprop(self.agent.tie.parameters(), lr = self.args.lr, alpha=0.97, eps=1e-6)


        self.n_nodes = int(self.n_agents * (self.n_agents - 1) / 2)
        self.interval = self.args.interval



    def optimizer_zero_grad(self):
        self.optimizer_agent_ac.zero_grad()
        self.optimizer_team_ac.zero_grad()


    def optimizer_step(self):
        self.optimizer_agent_ac.step()
        self.optimizer_team_ac.step()


    def compute_grad(self, batch):
        log=dict()
        agent_log = self.compute_agent_grad(batch[0])
        team_log = self.compute_team_grad(batch[1])

        merge_dict(agent_log, log)
        merge_dict(team_log, log)
        return log




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





    def collect_batch_data(self, batch_size):
        agent_batch_data = []
        team_batch_data = []
        batch_log = dict()
        num_episodes = 0

        while len(agent_batch_data) < batch_size:
            episode_data,episode_log = self.run_an_episode()
            agent_batch_data += episode_data[0]
            team_batch_data += episode_data[1]
            merge_dict(episode_log, batch_log)
            num_episodes += 1

        batch_data = Transition(*zip(*agent_batch_data))
        team_batch_data = Team_Transition(*zip(*team_batch_data))
        batch_data = [batch_data, team_batch_data]
        batch_log['num_episodes'] = num_episodes
        batch_log['num_steps'] = len(batch_data[0].actions)

        return batch_data, batch_log






    def run_an_episode(self):

        log = dict()

        memory = []
        team_memory = []

        self.reset()
        obs = self.env.get_obs()

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

        team_action_out, team_value = self.agent.clustering(obs_tensor)
        team_action = self.choose_action(team_action_out)

        rewards_list = []
        global_reward = np.zeros(1)

        step = 1
        num_group = 0
        episode_return = 0
        done = False

        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                team_action_out, team_value = self.agent.clustering(obs_tensor)
                team_action = self.choose_action(team_action_out)

            sets = self.matrix_to_set(team_action)
            after_comm = self.agent.communicate(obs_tensor, sets)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)

            rewards_list.append(np.mean(rewards).reshape(1))

            if step % self.interval == 0:
                global_reward = np.mean(rewards_list).reshape(1)

            next_obs = self.env.get_obs()

            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            global_episode_mask = np.ones(1)
            if done:
                episode_mask = np.zeros(rewards.shape)
                global_episode_mask = np.zeros(1)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)

            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)

            if step % self.interval == 0:
                 team_trans = Team_Transition(team_action_out, team_action, global_reward, team_value, global_episode_mask)
                 team_memory.append(team_trans)

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

        return (memory,team_memory), log





    def compute_team_grad(self, batch):

        log = dict()
        batch_size = len(batch.global_value)
        n = 1

        rewards = torch.Tensor(np.array(batch.global_reward))
        actions = torch.Tensor(np.array(batch.team_actions))
        actions = actions.transpose(1, 2).view(-1, n, 1)

        episode_masks = torch.Tensor(np.array(batch.episode_masks))

        values = torch.cat(batch.global_value, dim=0)
        action_outs = torch.stack(batch.team_action_outs, dim=0)

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

        log_p_a = [action_outs.view(-1, self.n_agents)]
        actions = actions.contiguous().view(-1, self.n_agents)
        log_prob = multinomials_log_density(actions, log_p_a)
        action_loss = -advantages.view(-1) * log_prob.squeeze()
        actor_loss = action_loss.sum()

        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        critic_loss = value_loss.sum()

        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()

        log['team_action_loss'] = actor_loss.item()
        log['team_value_loss'] = critic_loss.item()
        log['team_total_loss'] = total_loss.item()


        return log




