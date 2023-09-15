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
Team_Transition = namedtuple('Team_Transition', ('team_action_outs', 'team_actions', 'score'))

class RunnerHiercommCompetition(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)


        self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr = self.args.lr, alpha=0.97, eps=1e-6)
        self.optimizer_team_ac = RMSprop(self.agent.teaming.parameters(), lr = self.args.lr, alpha=0.97, eps=1e-6)


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

        team_action_out, team_value = self.agent.teaming(obs_tensor)
        team_action = self.choose_action(team_action_out)
        sets = self.matrix_to_set(team_action)
        score = self.get_score(sets, obs_tensor)

        step = 1
        num_group = 0
        episode_return = 0
        done = False

        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                team_action_out, team_value = self.agent.teaming(obs_tensor)
                team_action = self.choose_action(team_action_out)
                sets = self.matrix_to_set(team_action)
                score = self.get_score(sets, obs_tensor)



            after_comm = self.agent.communicate(obs_tensor, sets)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)


            next_obs = self.env.get_obs()

            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)

            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)

            if step % self.interval == 0:
                 team_trans = Team_Transition(team_action_out, team_action, score)
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
        n = 1

        actions = torch.Tensor(np.array(batch.team_actions))
        actions = actions.transpose(1, 2).view(-1, n, 1)
        action_outs = torch.stack(batch.team_action_outs, dim=0)
        score = torch.Tensor(np.array(batch.score)).view(-1, 1)



        log_p_a = [action_outs.view(-1, self.n_agents)]
        actions = actions.contiguous().view(-1, self.n_agents)
        log_prob = multinomials_log_density(actions, log_p_a)


        action_loss = - score.view(-1) * log_prob.squeeze()
        actor_loss = action_loss.sum()



        total_loss = actor_loss
        total_loss.backward()

        log['team_action_loss'] = actor_loss.item()
        log['team_value_loss'] = score.sum().item()
        log['team_total_loss'] = actor_loss.item()


        return log

    def get_score(self, sets, obs_tensor):
        # Compute similarity matrix
        matrix = self.cosine_similarity_matrix(obs_tensor)

        # Convert back to numpy for the remaining operations
        matrix = matrix.cpu().numpy()

        # Compute the modularity score
        m = np.sum(matrix) / 2.0
        score = 0
        for agent_set in sets:
            k_i = np.sum(matrix[agent_set, :])
            sum_of_edges = np.sum(matrix[np.ix_(agent_set, agent_set)])
            score += sum_of_edges - (k_i ** 2) / (4 * m)

        score = score / (2 * m)
        return score

    def cosine_similarity_matrix(self, obs):
        """
        obs: [n_agents, obs_dim] as a PyTorch tensor
        Returns a matrix of size [n_agents, n_agents] with the cosine similarity between rows.
        """
        norm = obs.norm(p=2, dim=1, keepdim=True)
        obs_normalized = obs.div(norm)

        similarity_matrix = torch.mm(obs_normalized, obs_normalized.t())

        # Set diagonal to zero
        similarity_matrix.fill_diagonal_(0)

        return similarity_matrix






