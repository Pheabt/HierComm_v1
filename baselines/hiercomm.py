import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv,GCNConv
import networkx as nx
import argparse
from modules.graph import measure_strength
from torch_geometric.data import Data



class HierCommAgent(nn.Module):

    def __init__(self, agent_config):
        super(HierCommAgent, self).__init__()

        self.args = argparse.Namespace(**agent_config)
        self.seed = self.args.seed

        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size

        self.agent = AgentAC(self.args)
        self.tie = Tie(self.args)
        self.clustering = clustering(self.args)

        if hasattr(self.args, 'random_prob'):
            self.random_prob = self.args.random_prob

        self.block = self.args.block


    def agent_clustering(self, obs):
        cmatrix = self.clustering(obs)
        return cmatrix


    def cmatrix_to_set(self, cmatrix):
        # cmatrix is a soft clustering matrix
        # return a list of sets

        sets = [[] for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            #max_index = torch.argmax(cmatrix[i,:])
            # sample based on prob
            index = torch.multinomial(cmatrix[i,:], 1).item()
            sets[index].append(i)

        # remove empty sets
        sets = [s for s in sets if s != []]

        return sets

    def communicate(self, local_obs, sets):
        local_obs = self.tie.local_emb(local_obs)

        #do attention for each set, and then concat

        intra_obs = torch.zeros_like(local_obs)
        inter_obs = torch.zeros_like(local_obs)

        global_set = []
        for set in sets:
            if len(set) > 1:
                member_obs = local_obs[set,:]
                intra_obs[set,:] = self.tie.intra_com(member_obs)
                pooling = self.tie.pooling(member_obs)
                global_set.append(pooling)
            else:
                intra_obs[set,:] = local_obs[set,:]
                global_set.append(local_obs[set,:])

        inter_obs_input = torch.cat(global_set, dim=0)
        inter_obs_output = self.tie.inter_com(inter_obs_input)

        for index, set in enumerate(sets):
            if len(set) > 1:
                inter_obs[set,:] = inter_obs_output[index,:].repeat(len(set), 1)
            else:
                inter_obs[set,:] = inter_obs_output[index,:]



        if self.block == 'no':
            after_comm = torch.cat((local_obs,  inter_obs,  intra_obs), dim=-1)
        elif self.block == 'inter':
            after_comm = torch.cat((local_obs,  intra_obs, torch.rand_like(inter_obs)), dim=-1)
        elif self.block == 'intra':
            after_comm = torch.cat((local_obs,  inter_obs, torch.rand_like(intra_obs)), dim=-1)
        else:
            raise ValueError('block must be one of no, inter, intra')

        return after_comm









class Tie (nn.Module):
    def __init__(self, args):
        super(Tie, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.tanh = nn.Tanh()
        self.att_head = self.args.att_head

        self.local_fc = nn.Linear(self.args.obs_shape, self.hid_size)

        self.intra_attn = nn.MultiheadAttention(self.hid_size, num_heads=self.att_head, batch_first=True)

        self.inter_attn = nn.MultiheadAttention(self.hid_size, num_heads=self.att_head, batch_first=True)
        self.inter_fc2 = nn.Linear(self.hid_size * 2, self.hid_size)

        self.attset_fc = nn.Linear(self.hid_size, 1)


    def local_emb(self, input):
        return self.tanh(self.local_fc(input))

    def intra_com(self, input):
        x = input.unsqueeze(0)
        h, _ = self.intra_attn(x,x,x)
        return h.squeeze(0)

    def inter_com(self, input):
        x = input.unsqueeze(0)
        h, _ = self.inter_attn(x,x,x)
        return h.squeeze(0)

    def pooling(self, input):

        #score = self.attset_fc(input)
        score = F.softmax(self.attset_fc(input), dim=0)

        #zip the input to 1 * fixed size output based on score
        output = torch.sum(score * input, dim=0, keepdim=True)    # [1, 1, hid_size]
        return output









class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()

        self.fc_1 = nn.Linear(self.hid_size * 3, self.hid_size)
        self.fc_2 = nn.Linear(self.hid_size, self.hid_size)
        self.actor_head = nn.Linear(self.hid_size, self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)



    def forward(self, final_obs):
        h = self.tanh(self.fc_1(final_obs))
        h = self.tanh(self.fc_2(h))
        a = F.log_softmax(self.actor_head(h), dim=-1)
        v = self.value_head(h)

        return a, v







class clustering(nn.Module):

    def __init__(self, args):
        super(clustering, self).__init__()

        self.args = args

        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size

        self.att_head = self.args.att_head




        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.args.obs_shape, self.hid_size)
        self.attn = nn.MultiheadAttention(self.hid_size, num_heads=self.att_head, batch_first=True)
        self.fc2 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.head = nn.Linear(self.hid_size , self.n_agents)



    def forward(self, x):

        x = self.tanh(self.fc1(x)).unsqueeze(0)
        h, _ = self.attn(x,x,x)

        xh = torch.cat([x.squeeze(0),h.squeeze(0)], dim=-1)

        z = self.tanh(self.fc2(xh))
        cmatrix = F.softmax(self.head(z), dim=-1)

        return cmatrix

