import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from .runner import Runner
from modules.utils import merge_dict, multinomials_log_density


Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards',
                                       'episode_masks', 'episode_agent_masks', 'values'))



class RunnerDefault(Runner):
    def __init__(self, config, env, agent):
        super(RunnerDefault, self).__init__(config, env, agent)
        # self.interval = self.args.interval
        # self.treshold = self.args.treshold
        self.treshold = -1
        self.interval = self.args.interval






    def run_an_episode(self):

        memory = []
        log = dict()
        episode_return = np.zeros(self.n_agents)

        self.reset()
        obs = self.env.get_obs()

        step = 1
        done = False
        num_group = 0


        graph = self.env.get_graph()
        g, set = self.agent.graph_partition(graph, self.treshold)

        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                graph = self.env.get_graph()
                g, set = self.agent.graph_partition(graph, self.treshold)

            after_comm = self.agent.communicate(obs_tensor, g, set)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)
            next_obs = self.env.get_obs()

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)

            trans = Transition(action_outs, actions, rewards,episode_mask, episode_agent_mask, values)
            memory.append(trans)


            obs = next_obs
            episode_return += rewards.astype(episode_return.dtype)
            step += 1
            num_group += len(set[1])


        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]
        log['num_groups'] = num_group / (step-1)

        if 'num_collisions' in env_info:
            log['num_collisions'] = env_info['num_collisions']

        # if self.args.env == 'tj':
        #     merge_dict(self.env.get_stat(), log)

        return memory ,log