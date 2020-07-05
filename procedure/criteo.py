from ignite.engine import Engine 
from ignite.engine import Events 
from ignite.metrics import Accuracy
from ignite.metrics import Loss 
from ignite.contrib.metrics import ROC_AUC 
from ignite.contrib.handlers.tqdm_logger import ProgressBar 

import torch 
from torch.utils.data import DataLoader 
from torch import nn 
from torch import optim 

from typing import Tuple 
from typing import Callable 

from argparse import Namespace 

def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
    feature_idx, feature_value, label = batch 
    return (
        feature_idx.to(device, non_blocking=True), 
        feature_value.to(device, non_blocking=True), 
        label.to(device, non_blocking=True)
    )


def Trainer(model: nn.Module, optimizer: optim.Optimizer, criterion: Callable, device: torch.device) -> Engine:  
    def _step(engine: Engine, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Namespace: 
        model.train() 
        with torch.enable_grad(): 
            feature_idx, feature_value, label = prepare_batch(batch, device=device) 
            logit = model(feature_idx, feature_value).squeeze() 
            loss = criterion(logit, label.float()) 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()  
        return Namespace(
            loss=loss.detach().cpu() 
        ) 
    return Engine(_step) 


def Evaluator(model: nn.Module, criterion: Callable, device: torch.device) -> Engine: 
    def _step(engine: Engine, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Namespace: 
        model.eval() 
        with torch.no_grad(): 
            feature_idx, feature_value, label = prepare_batch(batch, device=device) 
            logit = model(feature_idx, feature_value).squeeze() 
        return Namespace(
            logit=logit.cpu(), 
            label=label.cpu() 
        ) 
    return Engine(_step) 


def set_handlers(trainer: Engine, evaluator: Engine, valloader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, args: Namespace) -> None: 
    ROC_AUC(
        output_transform=lambda output: (output.logit, output.label)
    ).attach(engine=evaluator, name='roc_auc') 
    Accuracy(
        output_transform=lambda output: ((output.logit > 0).long(), output.label)
    ).attach(engine=evaluator, name='accuracy') 
    Loss(
        loss_fn=nn.BCEWithLogitsLoss(), 
        output_transform=lambda output: (output.logit, output.label.float())
    ).attach(engine=evaluator, name='loss') 

    ProgressBar(persist=True, desc='Epoch').attach(engine=trainer, output_transform=lambda output: {'loss': output.loss})
    ProgressBar(persist=False, desc='Eval').attach(engine=evaluator)
    ProgressBar(persist=True, desc='Eval').attach(
        engine=evaluator, 
        metric_names=['roc_auc', 'accuracy', 'loss'], 
        event_name=Events.EPOCH_COMPLETED, 
        closing_event_name=Events.COMPLETED 
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=args.evaluation_interval)) 
    def _evaluate(trainer: Engine): 
        evaluator.run(valloader, max_epochs=1) 
    






