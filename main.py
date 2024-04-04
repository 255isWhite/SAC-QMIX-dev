from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import torch
import sys
import os
from SAC.experiment import Experiment

class Args:
    pass


if __name__ == "__main__":

    
    config_path='config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    args = Args()
    #从json赋值给args
    args.__dict__.update(config)

    args.continue_run = False
    argv = sys.argv
    if len(argv)>1:
        if argv[1] == '-c':  ##continue
            args.continue_run = True

    #实例化Experiment
    experiment = Experiment(args)
    experiment.start()
    experiment.run()
    

    
