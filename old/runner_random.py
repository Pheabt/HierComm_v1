import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from .runner import Runner
from modules.utils import merge_dict, multinomials_log_density


Transition = namedtuple('Transition', ('obs', 'action_outs', 'actions', 'rewards',
                                       'episode_masks', 'episode_agent_masks', 'values'))



class RunnerRandom(Runner):
    def __init__(self, config, env, agent):
        super(RunnerRandom, self).__init__(config, env, agent)
        self.interval = self.args.interval




    def run_an_episode(self):

        memory = []
        log = dict()
        episode_return = np.zeros(self.n_agents)

        self.reset()
        obs = self.env.get_obs()


        step = 1
        done = False
        set = self.agent.random_set()
        num_group = 0
        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                set = self.agent.random_set()
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
            num_group += len(set)


        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]
        log['num_groups'] = num_group / (step-1)

        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(), log)


        return memory ,log