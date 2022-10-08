import yaml
import importlib
import inspect
from collections import defaultdict
from copy import deepcopy
import ast
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import itertools
import pandas as pd
import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

def initialize_obj(classname, args_dict=None):
    module_name, class_name = classname.rsplit(".", 1)
    Class = getattr(importlib.import_module(module_name), class_name)
    # filter by argnames
    if args_dict is not None:
        argspec = inspect.getfullargspec(Class.__init__)
        argnames = argspec.args
        args_dict = {k: v for k, v in args_dict.items()
                     if k in argnames or argspec.varkw is not None}

        defaults = argspec.defaults
        # add defaults
        if defaults is not None:
            for argname, default in zip(argnames[-len(defaults):], defaults):
                if argname not in args_dict:
                    args_dict[argname] = default
        class_instance = Class(**args_dict)
    else:
        class_instance = Class()
    return class_instance


def initialize(obj_config, update_args=None): # obj_config: dict; update_args: dict
    classname = obj_config['classname']
    kwargs = obj_config.get('args')
    if kwargs is None:
        kwargs = {}
    if update_args is not None:
        kwargs.update(update_args)
    return initialize_obj(classname, kwargs)

def init_transform(config, transform_type): # chain things in train_transforms or test_transforms together
    if transform_type + '_transforms' not in config:
        return None
    config_transforms = config[transform_type + '_transforms'] # train_transforms or test_transforms
    transform_list = [initialize(trans) for trans in config_transforms]
    return transforms.Compose(transform_list)

def init_dataset(config, dataset_type, template_dataset=None):
    '''
    Initializes a PyTorch Dataset for train, eval_train, validation, or test.
    Returns: torch.utils.data.Dataset
    '''
    
    custom_type = False
    transform_type = dataset_type
    if dataset_type in {'eval_train', 'val', 'test2'} or custom_type:
        transform_type = 'test'  # Use test transforms for eval sets.

    transform = init_transform(config, transform_type)
    target_transform = init_transform(config, transform_type + '_target')

    split_type = dataset_type
    # if dataset_type == 'eval_train':
    #     split_type = 'train'  # Default eval_train split is 'train'.
    dataset_kwargs = {'split': split_type, 'transform': transform,
                      'target_transform': target_transform,
                      'template_dataset': template_dataset,
                      'eval_mode': (dataset_type != 'train')}

    # if dataset_type == 'eval_train':  # Start off with args in 'train_args'.
    #     dataset_kwargs.update(config['dataset'].get('train_args', {}))
    dataset_kwargs.update(config['dataset'].get(dataset_type + '_args', {}))

    # We make a copy since the initialize function calls dict.update().
    dataset_config = deepcopy(config['dataset'])
    dataset = initialize(dataset_config, dataset_kwargs) # Key,这一步创建了dataset对象

    if config['dataset'].get('args', {}).get('standardize'):
        if dataset_type == 'train':  # Save training set's mean/std.
            config['dataset']['mean'] = dataset.get_mean()
            config['dataset']['std'] = dataset.get_std()
        else:  # Update dataset with training set's mean and std.
            dataset.set_mean(config['dataset']['mean'])
            dataset.set_std(config['dataset']['std'])

    if config['dataset'].get('args', {}).get('standardize_output'):
        if dataset_type == 'train':  # Save training set's output mean/std.
            config['dataset']['output_mean'] = dataset.get_output_mean()
            config['dataset']['output_std'] = dataset.get_output_std()
        else:  # Update dataset with training set's output mean and std.
            dataset.set_output_mean(config['dataset']['output_mean'])
            dataset.set_output_std(config['dataset']['output_std'])

    return dataset


def batch_loader(dataset, shuffle=True):
    data, targets, domains,lat_lons = dataset.data, dataset.targets, dataset.domain_labels, dataset.lat_lon

    if shuffle:
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)
        data, targets, domains, lat_lons = data[idxs], targets[idxs], domains[idxs], lat_lons[idxs]
    return itertools.cycle([{'data': data, 'target': targets, 'domain_label': domains, 'lat_lon': lat_lons}])

class DataParallel(torch.nn.DataParallel):
    '''
    Pass-through the attributes of the model thru DataParallel
    '''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def init_dataloader(config, dataset, dataset_type, shuffle=True):
    '''
    Initializes a PyTorch DataLoader around a provided dataset. Allows for
    specifying additional arguments via a config file, such as specifying the
    Sampler to use.
    '''
    if dataset_type not in ['train', 'eval_train', 'val', 'test', 'test2']:
        raise ValueError('{} is an invalid dataset type!'.format(dataset_type))
    dl_kwargs = {}
    if config['use_cuda']:
        dl_kwargs = {'num_workers': 2, 'pin_memory': True}
    batch_size = config.get('batch_size', 256)
    if dataset_type != 'train':
        batch_size = config.get('eval_batch_size', 256)
    dl_kwargs = {'batch_size': batch_size, 'shuffle': shuffle}

    if 'dataloader' in config:
        sampler = None
        dataloader_args = config['dataloader'].get('args', {})
        if 'sampler' in dataloader_args:
            sampler_kwargs = {'data_source': dataset}
            sampler = initialize(dataloader_args['sampler'], sampler_kwargs)
        dl_kwargs.update(dataloader_args)
        dataloader_args = config['dataloader'].get(dataset_type + '_args', {})
        if 'sampler' in dataloader_args:
            sampler_kwargs = {'data_source': dataset}
            sampler = initialize(dataloader_args['sampler'], sampler_kwargs)
        dl_kwargs.update(dataloader_args)
        dl_kwargs['sampler'] = sampler

    return DataLoader(dataset, drop_last=True, **dl_kwargs)

class CosineLR(LambdaLR):

    def __init__(self, optimizer, lr, num_epochs, offset=1):
        self.init_lr = lr
        fn = lambda epoch: lr * 0.5 * (1 + np.cos((epoch - offset) / num_epochs * np.pi))
        super().__init__(optimizer, lr_lambda=fn)

    def reset(self, epoch, num_epochs):
        self.__init__(self.optimizer, self.init_lr, num_epochs, offset=epoch)
