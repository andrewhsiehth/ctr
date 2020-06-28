from tqdm.auto import tqdm 

from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score

import torch 

def prepare_batch(batch, device):  
    record, label = batch 
    assert label.dim() == 2 and record.dim() == 2, f'{record.shape}, {label.shape}'
    return (record.to(device), label.to(device)) 

def train(model, dataloader, optimizer, criterion, device): 
    model.train() 
    with tqdm(dataloader, desc='[Train]', position=1, leave=True) as pbar: 
        with torch.enable_grad(): 
            for batch in pbar: 
                record, label = prepare_batch(batch, device)  
                logit = model(record) 
                loss = criterion(logit, label) 
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 
                pbar.set_postfix(loss=f'{loss.detach().item():.4f}') 

def evaluate(model, dataloader, criterion, device): 
    model.eval() 
    losses = torch.zeros((len(dataloader),)) 
    labels = torch.BoolTensor(size=(len(dataloader.dataset),))
    logits = torch.zeros((len(dataloader.dataset),)) 
    with tqdm(range(len(dataloader)), desc='[Eval]', position=1, leave=True) as pbar: 
        with torch.no_grad(): 
            for idx_batch, batch in enumerate(dataloader): 
                record, label = prepare_batch(batch, device)
                logit = model(record) 
                loss = criterion(logit, label) 
                losses[idx_batch:idx_batch+1] = loss.cpu()  
                labels[idx_batch*dataloader.batch_size:(idx_batch+1)*dataloader.batch_size] = label.squeeze().cpu() 
                logits[idx_batch*dataloader.batch_size:(idx_batch+1)*dataloader.batch_size] = logit.squeeze().cpu() 
                pbar.update(n=1)  
        accuracy = accuracy_score(y_true=labels.numpy(), y_pred=(logits > 0).numpy()) 
        roc_auc = roc_auc_score(y_true=labels.numpy(), y_score=logits.numpy()) 
        loss = torch.mean(losses).item()
        pbar.set_postfix(
            auc=f'{roc_auc:.4f}',
            acc=f'{accuracy:.4f}',  
            loss=f'{loss:.4f}'
        )
    return roc_auc, accuracy, loss  

        




                
        
