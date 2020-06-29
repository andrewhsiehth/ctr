from dataset import Criteo 
from model import DeepFM 
from procedure import train 
from procedure import evaluate 

import torch 
from torch import nn 
from torch import distributed 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel 

from tqdm.auto import tqdm 

from argparse import ArgumentParser 

import os 

def parse_args(): 
    parser = ArgumentParser() 
    parser.add_argument('--dataset_root', type=str, required=True) 
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count()) 
    parser.add_argument('--batch_size', type=int, default=2048) 
    parser.add_argument('--min_threshold', type=int, default=10) 
    parser.add_argument('--backend', type=str, default=distributed.Backend.NCCL)
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--world_size', type=int, default=1) 
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=10)
    parser.add_argument('--out_features', type=int, default=1)
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[400, 400, 400])
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[0.5, 0.5, 0.5])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda:0'))
    args, extra = parser.parse_known_args() 
    return args 


if __name__ == '__main__': 
    print('[parse args]') 
    args = parse_args() 
    print(args) 

    print('[prepare data]') 
    trainset, testset = Criteo.prepare_Criteo(
        root=args.dataset_root, 
        min_threshold=args.min_threshold, 
        n_jobs=os.cpu_count() 
    )  

    print('[init process group]') 
    # distributed.init_process_group(
    #     backend=args.backend, 
    #     init_method=args.init_method, 
    #     world_size=args.world_size, 
    #     rank=args.rank 
    # )
    torch.manual_seed(args.seed) 

    print('[init dataloader]') 
    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=False
    )
    # trainloader = DataLoader(
    #     dataset=trainset, 
    #     batch_size=args.batch_size, 
    #     sampler=DistributedSampler(trainset), 
    #     num_workers=args.num_workers
    # )
    testloader = DataLoader(
        dataset=testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        drop_last=False
    )

    print('[init model]') 
    model = nn.DataParallel(
        module=DeepFM(
            field_dims=trainset.field_dims, 
            embedding_dim=args.embedding_dim, 
            out_features=args.out_features, 
            hidden_units=args.hidden_units, 
            dropout_rates=args.dropout_rates 
        ).to(args.device) 
    ) 
    # model = DistributedDataParallel(DeepFM(
    #     field_dims=dataset.field_dims, 
    #     embedding_dim=args.embedding_dim, 
    #     out_features=args.out_features, 
    #     hidden_units=args.hidden_units, 
    #     dropout_rates=args.dropout_rates 
    # ).to(args.device))

    print('[init optimizer]') 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

    print('[init criterion]') 
    criterion = torch.nn.BCEWithLogitsLoss() 

    print('[start triaing]') 
    best_acc = 0.0 
    best_roc_auc = 0.0 
    with tqdm(range(args.n_epochs), desc='[Epoch]', position=0, leave=True, disable=('DISABLE_TQDM' in os.environ)) as pbar: 
        for epoch in pbar:
            # trainloader.sampler.set_epoch(epoch) 
            train(
                model=model, 
                dataloader=trainloader, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=args.device 
            ) 
            if epoch == 0: # force save before oom
                torch.save(model.module, os.path.join(args.checkpoint_dir, 'best.pt'))
            roc_auc, accuracy, loss = evaluate(
                model=model, 
                dataloader=testloader, 
                criterion=criterion, 
                device=args.device 
            )
            if roc_auc > best_roc_auc: 
                torch.save(model.module, os.path.join(args.checkpoint_dir, 'best.pt'))
            best_acc = max(best_acc, accuracy) 
            best_roc_auc = max(best_roc_auc, roc_auc) 
            pbar.set_postfix(acc=f'{best_acc:.4f}', roc_auc=f'{best_roc_auc:.4f}') 

    print('[destroy process group]') 
    # distributed.destroy_process_group() 

    print('[done]')


