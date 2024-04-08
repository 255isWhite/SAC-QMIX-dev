import torch
import numpy as np


class EpisodeBuffer:
    def __init__(self, scheme, args):
        self.scheme = scheme.copy()
        self.buffer_size = args.buffer_size
        self.max_seq_length = args.episode_limit
        # self.device = args.device
        self.device = 'cpu'
        self._setup_data()

    def _setup_data(self):
        self.index_st = 0
        self.n_sample = 0

        self.data = {}
        for k, v in self.scheme.items():
            shape = (self.buffer_size, self.max_seq_length) + v['shape']
            self.data[k] = torch.zeros(shape, dtype=v['dtype'],device=self.device)

    def sample(self, batch_size, top_n = 0):
        #当replay buffer中数据不足时，sample无法采样
        if self.n_sample < batch_size:
            return None

        #对id进行处理
        def trans_ids(ep_id):
            ep_id += self.index_st
            ep_id %= self.buffer_size
            return ep_id

        #随机数范围[0，n_sample-1]
        #top_n 表示最新的top_n个episode一定会选取
        ep_ids = np.random.choice(self.n_sample - top_n, batch_size - top_n, replace=False)
        ep_ids = list(ep_ids) + [self.n_sample-i-1 for i in range(top_n)]# [self.n_sample -1]
        ep_ids = list(map(trans_ids, ep_ids)) #len(ep_lids) = batch_size

        ret = {}
        for k, v in self.data.items():
            ret[k] = v[ep_ids]

        return ret #len(ret) = bathc_size
    

    def add_episode(self, data):

        #len_ep = len
        if self.n_sample < self.buffer_size:
            self.n_sample += 1. #[0,self.buffer_size]
        else:
            #self.index_st代表第一个buffer_size达成后的data数量
            self.index_st += 1
            self.index_st %= self.buffer_size

        index_ep = (self.index_st + self.n_sample - 1) % self.buffer_size
        
        for k, v in data.items():

            ep_len = len(v)
            dtype = self.scheme[k]['dtype']
            #why this
            self.data[k][index_ep].zero_()
            self.data[k][index_ep, 0:ep_len] = torch.as_tensor(v, dtype=dtype, device=self.device)

    def clear(self):
        self.index_st = 0
        self.n_sample = 0
        for item in self.data.values():
            item.zero_()
        

    def state_dict(self):
        buffer_state = {'index_st': self.index_st, 'n_sample': self.n_sample, 'data': self.data}
        return buffer_state
    
    def load_state_dict(self, state_dict):
        self.index_st = state_dict['index_st']
        self.n_sample = state_dict['n_sample']    
        self.data = state_dict['data']
