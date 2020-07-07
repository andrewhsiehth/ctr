from dataset import LIBSVMDataset 
from model import DeepFM 

from procedure.criteo import Trainer 
from procedure.criteo import Evaluator 
from procedure.criteo import set_handlers 

import torch 
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader 

from argparse import ArgumentParser 


def parse_args(): 
    parser = ArgumentParser() 
    parser.add_argument('--data_uri_train', type=str, required=True) 
    parser.add_argument('--data_uri_val', type=str, default=None) 
    parser.add_argument('--data_uri_test', type=str, default=None) 
    parser.add_argument('--checkpoint_dir', type=str, required=True) 
    parser.add_argument('--device', type=torch.device, default=torch.device('cpu')) 
    parser.add_argument('--batch_size', type=int, default=2048) 
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_fields', type=int, default=39)
    parser.add_argument('--num_features', type=int, default=117581)
    parser.add_argument('--embedding_dim', type=int, default=256) 
    parser.add_argument('--out_features', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=5e-4) 
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[400, 400, 400]) 
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[0.3, 0.3, 0.3]) 
    parser.add_argument('--evaluation_interval', type=int, default=200) 
    parser.add_argument('--max_epochs', type=int, default=5) 

    args, extra = parser.parse_known_args() 
    return args 


if __name__ == '__main__': 
    print('[parse args]')
    args = parse_args() 
    print(args) 

    print('[prepare data]') 
    trainset = LIBSVMDataset.prepare_dataset(data_uri=args.data_uri_train) 
    valset = LIBSVMDataset.prepare_dataset(data_uri=args.data_uri_val) 

    print('[init dataloader]') 
    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=False 
    )
    valloader = DataLoader(
        dataset=valset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        drop_last=False 
    )

    print('[init model]') 
    model = DeepFM(
        num_fields=args.num_fields, 
        num_features=args.num_features, 
        embedding_dim=args.embedding_dim, 
        out_features=args.out_features, 
        hidden_units=args.hidden_units, 
        dropout_rates=args.dropout_rates  
    )

    print('[init optimizer]') 
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 

    print('[init criterion]') 
    criterion = nn.BCEWithLogitsLoss() 

    print('[init engines]') 
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=args.device 
    ) 
    evaluator = Evaluator(
        model=model, 
        criterion=criterion, 
        device=args.device 
    )

    print('[set handlers]')
    set_handlers(
        trainer=trainer, 
        evaluator=evaluator, 
        valloader=valloader, 
        model=model, 
        optimizer=optimizer, 
        args=args 
    )


    print('[start training]') 
    trainer.run(trainloader, max_epochs=args.max_epochs) 

    print('[done]')
