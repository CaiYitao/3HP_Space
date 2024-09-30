
from typing import Iterable, Union
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch

def choose_optimizer(parameters: Iterable[torch.Tensor], optimizer_type: str, **kwargs) -> optim.Optimizer:
    """
    Choose an optimizer based on the provided type.

    Parameters:
        parameters (Iterable[torch.Tensor]): Model parameters to optimize.
        optimizer_type (str): Type of optimizer to use.
        **kwargs: Additional keyword arguments for the optimizer.

    Returns:
        optim.Optimizer: Selected optimizer instance.
    """
    optimizer_map = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'asgd': optim.ASGD,
        'adagrad': optim.Adagrad
    }
    
    optimizer_class = optimizer_map.get(optimizer_type.lower())
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer type. Choose from {', '.join(optimizer_map.keys())}.")
    
    return optimizer_class(parameters, **kwargs)


def choose_lr_scheduler(optimizer: optim.Optimizer, scheduler_type: str, **kwargs) -> Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]:
    """
    Choose a learning rate scheduler based on the provided type.

    Parameters:
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler_type (str): Type of learning rate scheduler.
        **kwargs: Additional keyword arguments specific to the chosen scheduler.

    Returns:
        Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]: Selected learning rate scheduler instance.
    """
    scheduler_map = {
        'steplr': lr_scheduler.StepLR,
        'multisteplr': lr_scheduler.MultiStepLR,
        'exponentiallr': lr_scheduler.ExponentialLR,
        'reducelronplateau': lr_scheduler.ReduceLROnPlateau,
        'lambdalr': lr_scheduler.LambdaLR,
        'cosineannealingwarmrestarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'cosineannealinglr': lr_scheduler.CosineAnnealingLR,
        'cycliclr': lr_scheduler.CyclicLR,
        'onecyclelr': lr_scheduler.OneCycleLR
    }
    
    scheduler_class = scheduler_map.get(scheduler_type.lower())
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler type. Choose from {', '.join(scheduler_map.keys())}.")
    
    return scheduler_class(optimizer, **kwargs)
