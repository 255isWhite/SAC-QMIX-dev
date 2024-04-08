import torch
import torch.nn as nn
import torch.nn.functional as F
from .actor import Actor

class Actors(nn.Module):
    def __init__(self, args):
        super(Actors, self).__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        #单智能体
        self.agent = Actor(args)

    def init_hiddens(self, n_batch):
        hiddens = self.agent.init_hidden(n_batch*self.n_agents)
        hiddens = hiddens.reshape(n_batch,self.n_agents,-1)
        return hiddens

    def forward(self, states, avail_actions, hiddens):
        n_batch = states.shape[0]
        
        #确保二维tensor
        states = states.reshape(-1,states.shape[-1])
        hiddens = hiddens.reshape(-1,hiddens.shape[-1])
        avail_actions = avail_actions.reshape(-1,avail_actions.shape[-1])
        
        #网络forward
        ys, hs_next = self.agent.forward(states, avail_actions, hiddens) #hiddens体现了多个agent
        
        #shape [n_batch,n_agents,n_actions]
        ys = ys.reshape(n_batch,self.n_agents,-1)
        hs_next = hs_next.reshape(n_batch,self.n_agents,-1)
        
        return ys, hs_next

