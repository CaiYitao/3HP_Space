import random
from collections import deque
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd

import torch
import numpy as np

class Transition:
    def __init__(self, state, action, reward, log_probs):
        self.state = state
        self.action = action
        self.reward = reward
        self.g_return = 0.0
        self.log_probs = log_probs


class Episode:
    def __init__(self, discount):
        self.discount = discount
        self._empty()
        self.total_reward = 0.0

    def _empty(self):
        self.n = 0
        self.transitions = []

    def reset(self):
        self._empty()

    def size(self):
        return self.n

    def append(self, transition):
        self.transitions.append(transition)
        self.n += 1
        
    def states(self):
        return [s.state for s in self.transitions]
    
    def actions(self):
        return [a.action for a in self.transitions]
    
    def rewards(self):
        return [r.reward for r in self.transitions]
    
    def returns(self):
        return [r.g_return for r in self.transitions]
    
    def calculate_return(self):
        # turn rewards into return
        rewards = self.rewards()
        trajectory_len = len(rewards)
        return_array = torch.zeros((trajectory_len,))
        g_return = 0.
        for i in range(trajectory_len-1, -1, -1):
            g_return = rewards[i] + self.discount * g_return
            return_array[i] = g_return
            self.transitions[i].g_return = g_return
        return return_array

class ReplayBuffer:
    def __init__(self, config):
        self.capacity = config.buffer_capacity
        self.batch_size = config.batch_size
        # self.min_transitions = min_transitions
        self.buffer = []
        self._empty()
        self.mean_returns = []
        self.all_returns = []
        
    def _empty(self):
        del self.buffer[:]
        self.position = 0

    def add(self, episode):
        """Saves a transition."""
        episode.calculate_return()
        for t in episode.transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = t
            self.position = (self.position + 1) % self.capacity
            print(f"len: {len(self.buffer)}")
            
    def update_stats(self):
        returns = [t.g_return for t in self.buffer]
        self.all_returns += returns
        mean_return = np.mean(np.array(returns))
        self.mean_returns += ([mean_return]*len(returns))

    def reset(self):
        self._empty()

    def sample(self):
        if len(self.buffer) < self.batch_size:
        # If the buffer size is smaller than the batch size,
        # return all transitions in the buffer without replacement
            return self.buffer
        else:
            print(f"buffer sample: {self.buffer} buffer length: {len(self.buffer)}")
            prob = [1/len(self.buffer) for _ in range(0, len(self.buffer))]
            print(f"buffer sample prob: {prob}")
            return np.random.choice(self.buffer, size=self.batch_size, p=prob, replace=False)

    def __len__(self):
        return len(self.buffer)


def grad_variance(g):
    return np.mean(g**2) - np.mean(g)**2


class Logger:
    def __init__(self):
        self.gradients = []

    def add_gradients(self, grad):
        self.gradients.append(grad)

    def compute_gradient_variance(self):
        vars_ = []
        grads_list = [np.zeros_like(self.gradients[0])] * 100
        for i, grads in enumerate(self.gradients):
            grads_list.append(grads)
            grads_list = grads_list[1:]
            grad_arr = np.stack(grads_list, axis=0)
            g = np.apply_along_axis(grad_variance, axis=-1, arr=grad_arr)
            vars_.append(np.mean(g))
        return vars_


def choose_optimizer(parameters, optimizer_type,**kwargs):
    """
    Choose an optimizer based on the provided type.

    Parameters:
        parameters (iterable): Model parameters to optimize.
        optimizer_type (str): Type of optimizer to use ('sgd', 'adam', 'asgd', 'adagrad').
        lr (float): Learning rate for the optimizer.
        weight_decay (float, optional): Weight decay (L2 penalty). Default is None.

    Returns:
        optimizer: Selected optimizer instance.
    """
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(parameters, **kwargs)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(parameters, **kwargs)
    elif optimizer_type.lower() == 'asgd':
        optimizer = optim.ASGD(parameters, **kwargs)
    elif optimizer_type.lower() == 'adagrad':
        optimizer = optim.Adagrad(parameters, **kwargs)
    else:
        raise ValueError("Unsupported optimizer type. Choose from 'sgd', 'adam', 'asgd', or 'adagrad'.")

    return optimizer





def choose_lr_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Choose a learning rate scheduler based on the provided type.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler_type (str): Type of learning rate scheduler ('step', 'multi_step', 'exponential', 'reduce_on_plateau', 'lambda', 'cosine_annealing_warm_restarts', 'cosine_annealing_lr_decay').
        kwargs: Additional keyword arguments specific to the chosen scheduler.

    Returns:
        scheduler: Selected learning rate scheduler instance.
    """
    if scheduler_type.lower() == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'multisteplr':
        scheduler = lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'exponentiallr':
        scheduler = lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'reducelronplateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_type.lower() == 'lambdalr':
        scheduler = lr_scheduler.LambdaLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'cosineannealingwarmrestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    elif scheduler_type.lower() == 'cosineannealinglr':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'onecyclelr':
        scheduler = lr_scheduler.OneCycleLR(optimizer, **kwargs)
    else:
        raise ValueError("Unsupported scheduler type. Choose from available options.")

    return scheduler
