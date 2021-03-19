from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
#from .vdn_actor import VDNActor
#from .vdn_critic import VDNCritic
from datetime import datetime
#from .learner_q import Learner
#from .controller_q import Controller
from .learner_q import Learner
from .controller_q import Controller
from .episode_buffer import EpisodeBuffer
from .runnner import Runner
from writter_util import WritterUtil

class Experiment:
    def __init__(self, args):        
        self.args = args


    def save(self):
        run_state = {}
        run_state['args'] = self.args.__dict__.copy()
        del run_state['args']['new_run']
        run_state['model_state'] = self.sys_agent.state_dict()
        run_state['target_state'] = self.learner.sys_agent_tar.state_dict()
        run_state['optim_state'] = self.learner.optimizer.state_dict()
        run_state['episode'] = self.e            
        run_state['buffer_state'] = self.buffer.state_dict()
        torch.save(run_state, self.path_checkpt)


    def start(self):
        args = self.args
        path_checkpts = 'checkpoints'
        if not os.path.exists(path_checkpts):
            os.mkdir(path_checkpts)
        path_checkpt = os.path.join(path_checkpts, args.run_name+ '.tar')
        # if not args.new_run:
        #     run_state = torch.load(path_checkpt)
        #     args.__dict__.update(run_state['args'])            
            
        env = StarCraft2Env(map_name=args.map_name, window_size_x=640, window_size_y=480)
        env_info = env.get_env_info()

        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        
        args.n_agents = n_agents
        args.n_actions = n_actions
        args.obs_dim = env_info['obs_shape']
        args.input_dim = args.obs_dim
        if args.agent_id:
            args.input_dim += args.n_agents        
        args.state_dim = env_info['state_shape']
        args.episode_limit = env_info['episode_limit']

        # if args.actor_critic:
        #     if args.msg_onehot:
        #         from Agent.mq_ac2 import Controller,Learner,MQAgent              
        #     else:
        #         from Agent.mq_ac import Controller,Learner,MQAgent              
        # elif args.model_idp:
        #     from Agent.mq_idp import Controller,Learner,MQAgent
        # else:
        #     from learner import Learner
        #     from controller import Controller
        #     from Agent import VDNAgent,MQAgent,JALAgent
        

        # if args.model == 'mq':
        #     sys_agent = MQAgent(args)                    
        # elif args.model == 'jal':
        #     sys_agent = JALAgent(args)
        # else:
        #     sys_agent = VDNAgent(args)
        #sys_actor = VDNActor(args)
        #sys_critic = VDNCritic(args)
        #if args.device == 'cuda':        
            #sys_actor.cuda()
            #sys_critic.cuda()
        
        writter = SummaryWriter('runs/'+ args.run_name + '/' + datetime.now().strftime('%Y-%m-%d,%H%M%S'))
        w_util = WritterUtil(writter,args)
        learner = Learner(args)
        #sys_actor = learner.sys_actor        
        ctrler = Controller(learner.sys_critic,args)
        runner = Runner(env,ctrler,args)
        

        scheme = {}
        scheme['obs'] = {'shape':(n_agents,args.obs_dim), 'dtype': torch.float32}
        scheme['valid'] = {'shape':(), 'dtype': torch.int32}
        scheme['actions'] = {'shape':(n_agents,), 'dtype': torch.int32}
        scheme['avail_actions'] = {'shape':(n_agents, n_actions), 'dtype': torch.int32}
        scheme['reward'] = {'shape':(), 'dtype': torch.float32}
        scheme['state'] = {'shape':(args.state_dim,), 'dtype': torch.float32}

        buffer = EpisodeBuffer(scheme, args)
        
        e = 0
        if not args.new_run:
            run_state = torch.load(path_checkpt)
            e = run_state['episode'] + 1
            sys_agent.load_state_dict(run_state['model_state'])
            learner.sys_agent_tar.load_state_dict(run_state['target_state'])
            #learner.sys_agent_tar.load_state_dict(run_state['model_state'])
            learner.optimizer.load_state_dict(run_state['optim_state'])
            buffer.load_state_dict(run_state['buffer_state'])

        self.path_checkpt = path_checkpt
        self.e = e
        self.env = env
        #self.sys_actor = sys_actor
        #self.sys_critic = sys_critic
        self.writter = writter
        self.w_util = w_util
        self.ctrler = ctrler
        self.runner = runner
        self.learner = learner
        self.buffer = buffer
        

        
    def run(self):
        e = self.e
        args = self.args
        buffer = self.buffer
        #sys_agent = self.sys_agent
        runner = self.runner
        learner = self.learner
        w_util = self.w_util


        loss = None
        for e in range(e ,args.n_episodes):
                        
            data, episode_reward, _ =  runner.run()                    
            buffer.add_episode(data)
            data = buffer.sample(args.n_batch)        
            w_util.WriteScalar('train/reward', episode_reward, e)
            if data:
                learn_info = learner.train(data)                        
                w_util.WriteScalar('train/loss', learn_info['loss'], e)
                w_util.WriteScalar('train/q', learn_info['q'], e)
                #w_util.WriteScalar('train/h', learn_info['h'], e)
                #w_util.WriteScalar('train/alpha', learn_info['alpha'], e)
                # print("Episode {}, reward = {}， loss = {},  q= {},  h= {},  alpha = {}".format\
                #     (e, episode_reward, learn_info['loss'],learn_info['q'],learn_info['h'],learn_info['alpha']))
                print("Episode {}, reward = {}， loss = {},  q= {}".format\
                    (e, episode_reward, learn_info['loss'],learn_info['q']))
            else:
                print("Episode {}, reward = {}".format(e, episode_reward))
            if e % args.test_every == 0:
                win_count = 0
                reward_avg = 0
                for i in range(args.test_count):
                    _, episode_reward, win_tag =  runner.run(test_mode=True)
                    if win_tag:
                        win_count += 1
                    reward_avg += episode_reward                
                win_rate = win_count/args.test_count
                reward_avg /= args.test_count
                w_util.WriteScalar('test/reward', reward_avg, e)
                w_util.WriteScalar('test/win_rate', win_rate, e)
                
                print('Test reward = {}, win_rate = {}'.format(reward_avg, win_rate))

            #if e % args.log_every == 0:
                
                #w_util.WriteModel('model', sys_agent, e)
                #self.e = e
                #self.save()
        self.env.close()