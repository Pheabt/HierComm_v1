import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from .runner import Runner
from torch.optim import Adam,RMSprop
from modules.utils import merge_dict, multinomials_log_density


Transition = namedtuple('Transition', ('obs', 'action_outs', 'actions', 'rewards',
                                       'episode_masks', 'episode_agent_masks', 'values'))



class RunnerOne(Runner):
    def __init__(self, config, env, agent):
        super(RunnerOne, self).__init__(config, env, agent)

        self.onegroup = [list(range(self.n_agents))]
        self.no_group = [[i] for i in range(self.n_agents)]

        # self.params = [p for p in self.agent.agent.parameters()]
        # # self.optimizer_agent_ac = Adam(params=self.agent.parameters(), lr=self.args.lr)
        # self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr=self.args.lr, alpha=0.97, eps=1e-6)


    def run_an_episode(self):

        memory = []
        log = dict()
        episode_return = np.zeros(self.n_agents)

        self.reset()
        obs = self.env.get_obs()


        step = 1
        done = False
        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            set = self.onegroup
            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)
            next_obs = self.env.get_obs()

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            else:
                if 'is_completed' in env_info:
                    episode_agent_mask = 1 - env_info['is_completed'].reshape(-1)

            trans = Transition(np.array(obs), action_outs, actions, np.array(rewards),
                                    episode_mask, episode_agent_mask, values)
            memory.append(trans)


            obs = next_obs
            episode_return += rewards.astype(episode_return.dtype)
            step += 1


        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]

        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(), log)


        return memory ,log