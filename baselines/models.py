import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.utils import from_networkx
import argparse
from torch_geometric.data import Data


class MLP(nn.Module):
    def __init__(self, agent_config):
        super(MLP, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        self.hid_size = self.args.hid_size
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions

        self.affine1 = nn.Linear(self.obs_shape, self.hid_size)
        self.affine2 = nn.Linear(self.hid_size, self.hid_size)
        self.head = nn.Linear(self.hid_size,self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)
        self.tanh = nn.Tanh()


    def forward(self, x, info={}):
        x = self.tanh(self.affine1(x))
        h = self.tanh(self.affine2(x))
        # h = self.tanh(sum([self.affine2(x), x]))
        a = F.log_softmax(self.head(h), dim=-1)
        v = self.value_head(h)

        return a, v





class Attention(nn.Module):
    def __init__(self, agent_config):
        super(Attention, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        self.att_head = self.args.att_head
        self.hid_size = self.args.hid_size
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()


        self.affine1 = nn.Linear(self.obs_shape, self.hid_size)
        self.attn = nn.MultiheadAttention(self.hid_size, num_heads=self.att_head, batch_first=True)
        self.affine2 = nn.Linear(self.hid_size * 2, self.hid_size)

        self.head = nn.Linear(self.hid_size,self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)







    def forward(self, x, info={}):

        x = self.tanh(self.affine1(x)).unsqueeze(0)
        h, _ = self.attn(x,x,x)
        xh = torch.cat([x.squeeze(0),h.squeeze(0)], dim=-1)
        z = self.tanh(self.affine2(xh))
        a = F.log_softmax(self.head(z), dim=-1)
        v = self.value_head(z)
        return a, v

class Attention_Noise(nn.Module):
    def __init__(self, agent_config):
        super(Attention_Noise, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        self.att_head = self.args.att_head
        self.hid_size = self.args.hid_size
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()

        self.affine1 = nn.Linear(self.obs_shape, self.hid_size)
        self.attn = nn.MultiheadAttention(self.hid_size, num_heads=self.att_head, batch_first=True)
        self.affine2 = nn.Linear(self.hid_size * 2, self.hid_size)

        self.head = nn.Linear(self.hid_size,self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)


    def forward(self, x, info={}):

        x = self.tanh(self.affine1(x))
        xx = x[:6, :].unsqueeze(0)
        h, _ = self.attn(xx,xx,xx)
        h = torch.cat([xx.squeeze(0),h.squeeze(0)], dim=0)
        xh = torch.cat([x.squeeze(0),h.squeeze(0)], dim=-1)
        z = self.tanh(self.affine2(xh))
        a = F.log_softmax(self.head(z), dim=-1)
        v = self.value_head(z)
        return a, v


class GNN(nn.Module):
    def __init__(self, agent_config):
        super(GNN, self).__init__()
        self.args = argparse.Namespace(**agent_config)
        self.hid_size = self.args.hid_size
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()


        self.fnn1 = nn.Linear(self.obs_shape, self.hid_size)
        #self.conv2 = GATConv(self.hid_size, self.hid_size, heads=1)
        self.conv2 = GCNConv(self.hid_size, self.hid_size)
        # self.conv3 = GATConv(self.hid_size, self.hid_size, heads=1)
        # self.conv4 = GATConv(self.hid_size, self.hid_size, heads=1)
        self.fnn3 = nn.Linear(self.hid_size *2, self.hid_size)

        self.head = nn.Linear(self.hid_size ,self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)



    def forward(self, obs, g):

        x = self.tanh(self.fnn1(obs))
        if list(g.edges()) == []:
            h = x
        else:
            edge_index = torch.tensor(list(g.edges()), dtype=torch.long)
            data = Data(x=x,edge_index=edge_index.t().contiguous())
            h = self.tanh(self.conv2(data.x, data.edge_index))
        # data = Data(x=h, edge_index=edge_index.t().contiguous())
        # h = self.tanh(self.conv3(data.x, data.edge_index))
        # data = Data(x=h, edge_index=edge_index.t().contiguous())
        # h = self.tanh(self.conv4(data.x, data.edge_index))
        h = self.tanh(self.fnn3(torch.cat([x,h], dim=-1)))
        #h = self.tanh(self.fnn3(h))
        a = F.log_softmax(self.head(h), dim=-1)
        v = self.value_head(h)

        return a, v