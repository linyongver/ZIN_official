from .model import InferEnv
import torch
import pandas as pd

class ERM:
    
    def __init__(self, flags, dp):
        pass

    def __call__(self, batch_data, step,  mlp=None, scale=None, mean_nll=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        train_logits = mlp(train_x)
        train_nll = mean_nll(train_logits, train_y)
        train_penalty = torch.tensor(0.0)
        return train_nll, train_penalty

