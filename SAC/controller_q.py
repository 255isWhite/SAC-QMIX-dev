import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


#n_batch = 1


class Controller:
    def __init__(self, sys_agent, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions        
        #self.epsilon_st = args.epsilon_st
        self.epsilon_ed = args.epsilon_ed
        #self.eps_len = args.eps_len
        self.eps_delta = (args.epsilon_st - args.epsilon_ed)/args.eps_len
        self.epsilon = args.epsilon_st
        self.agent_id = args.agent_id
        self.device = args.device
        self.explore_type = args.explore_type
        self.sys_agent_src = sys_agent
        self.sys_agent = type(sys_agent)(args)        
        self.sys_agent.eval()
        self.episode = 0

    def new_episode(self):
        state_dict = self.sys_agent_src.state_dict()
        self.sys_agent.load_state_dict(state_dict)        
        self.hiddens = self.sys_agent.init_hiddens(1)        
        self.episode += 1                

    def get_actions(self, states, avail_actions, explore=False):
        
        self.epsilon -= self.eps_delta
        if self.epsilon < self.epsilon_ed:
            self.epsilon = self.epsilon_ed
        epsilon = self.epsilon
        
        states = torch.as_tensor(states).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions).unsqueeze(0)
        if self.agent_id:
            agent_ids = torch.eye(self.n_agents,device=states.device)
            agent_ids = agent_ids.reshape((1,)*(states.ndim-2)+agent_ids.shape)            
            agent_ids = agent_ids.expand(states.shape[:-2]+(-1,-1))
            states = torch.cat([states,agent_ids],-1)
        with torch.no_grad():
            qs, hs_next = self.sys_agent.forward(states,avail_actions, self.hiddens)
        self.hiddens = hs_next
        
        if explore:
            q_rands = torch.rand_like(qs)
            q_rands[avail_actions == 0] = -float('inf')
            eps_rands = torch.rand(qs.shape[:2])  #1, self.n_agents
            qs[eps_rands < epsilon] = q_rands[eps_rands < epsilon]
        
        actions = torch.argmax(qs[0],-1)
        return actions.numpy()

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)