from dataset import Criteo 

import torch 
from torch import distributed 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel 

from tqdm import tqdm 

from argparse import ArgumentParser 

import os 

def parse_args(): 
    parser = ArgumentParser() 
    parser.add_argument('--data_path', type=str, required=True) 
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--min_threshold', type=int, default=10) 
    parser.add_argument('--backend', type=str, default=distributed.Backend.NCCL)
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--world_size', type=int, default=1) 
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args, extra = parser.parse_known_args() 
    return args 


if __name__ == '__main__': 
    print('[parse args]') 
    args = parse_args() 
    print(args) 

    print('[prepare data]') 
    dataset = Criteo(data_path=args.data_path, min_threshold=args.min_threshold)  

    print('[init process group]') 
    distributed.init_process_group(
        backend=args.backend, 
        init_method=args.init_method, 
        world_size=args.world_size, 
        rank=args.rank 
    )
    torch.manual_seed(args.seed) 

    print('[init dataloader]') 
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        sampler=DistributedSampler(dataset), 
        num_workers=args.num_workers 
    )

    for batch in tqdm(dataloader): 
        print(batch) 
        # break

    print('[destroy process group]') 
    distributed.destroy_process_group() 


