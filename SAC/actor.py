import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args

        self.input_dim = args.input_dim        
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def init_hidden(self, n_batch):
        # make hidden states on same device as model
        return self.fc1.weight.new_zeros(n_batch, self.rnn_hidden_dim)

    def forward(self, inputs, avail_actions, h_last):
        
        # FC->ReLU->GRU->FC

        h = F.relu(self.fc1(inputs))        
        h = self.rnn(h, h_last)
        y = self.fc2(h)

        #unavailable actions are masked as inf
        y[avail_actions == 0] = -float('inf')
        
        # -1维度上softmax
        # y.shape = [n_batch, n_actions]
        y = torch.softmax(y,-1)

        return y,h
    

