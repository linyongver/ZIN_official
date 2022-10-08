
from collections import OrderedDict

from torch import nn

import torch
from collections import OrderedDict

from torch import nn
from torch.autograd import Variable
import numpy as np
import itertools

class MultitaskModel(nn.Module):
    '''
    A multilayer perceptron for learning multiple tasks simultaneously.
    The model consists of pairs of fully connected and ReLU layers.
    '''
    def __init__(self, feature_dim, task_dims, shared_layers, batch_norm=False, use_idx=None,
                 freeze_shared=False, dropout_prob=0.0):
        '''
        Constructor.

        Parameters
        ----------
        shared_dims : List[int]
            Defines the number and sizes of hidden layers that are shared
            amongst the tasks.
        task_dims : List[List[int]]
            Defines the number and sizes of hidden layers for a variable number of tasks.
        use_idx: int
            Use only a certain head
        freeze_shared: only make the heads trainable.
        '''
        super().__init__()
        self.use_idx = use_idx
        self.shared_layers = shared_layers
        self.freeze_shared = freeze_shared
        self.task_layers = nn.ModuleList()
        for i in range(len(task_dims)):
            curr_task_layers = []
            linear = nn.Linear(feature_dim, task_dims[i][0])
            curr_task_layers.append((f'Task{i + 1}Linear{1}', linear))
            for j in range(1, len(task_dims[i])):
                if batch_norm:
                    curr_task_layers.append((f'Task{i + 1}BN{j}', nn.BatchNorm1d(task_dims[i][j-1])))
                curr_task_layers.append((f'Task{i + 1}ReLU{j}', nn.ReLU()))
                linear = nn.Linear(task_dims[i][j - 1], task_dims[i][j])
                curr_task_layers.append((f'Task{i + 1}Linear{j + 1}', linear))
            curr_task_sequential = nn.Sequential(OrderedDict(curr_task_layers))
            self.task_layers.append(curr_task_sequential)

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

    def trainable_params(self):
        if self.freeze_shared:
            return itertools.chain(*[l.parameters() for l in self.task_layers])
        else:
            return self.parameters()

    def forward(self, x):
        if isinstance(x, list) and not self.training:
            x = x[0]

        if isinstance(x, list):
            intermed_outs = [self.shared_layers(xi) for xi in x]
            if self.dropout_prob > 0.0:
                intermed_outs = [self.dropout(out) for out in intermed_outs]
            return [layer(intermed_out) for layer, intermed_out in zip(self.task_layers, intermed_outs)]
        else:
            shared_output = self.shared_layers(x)
            if self.dropout_prob > 0.0:
                shared_output = self.dropout(shared_output)

            if self.use_idx is not None:
                return self.task_layers[self.use_idx](shared_output)

            if self.training:
                return [layer(shared_output) for layer in self.task_layers]
            else:  # For eval, return first task output only.
                return self.task_layers[0](shared_output)


class CNN1DFeatureExtractor(nn.Module):
    def __init__(self, in_channels, output_size=128, batch_norm=False):
        super().__init__()
        self.output_size = output_size
        self.in_channels = in_channels

        activ = nn.ReLU(True)

        if batch_norm:
            self.feature_extractor = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels, 32, 5, padding=2)),
                ('bn1', nn.BatchNorm1d(32)),
                ('relu1', activ),
                ('conv2', nn.Conv1d(32, 32, 3, padding=1)),
                ('bn2', nn.BatchNorm1d(32)),
                ('relu2', activ),
                ('maxpool1', nn.MaxPool1d(2, 2)),
                ('conv3', nn.Conv1d(32, 64, 3, padding=1)),
                ('bn3', nn.BatchNorm1d(64)),
                ('relu3', activ),
                ('maxpool2', nn.MaxPool1d(2, 2)),
                ('conv4', nn.Conv1d(64, output_size, 3, padding=1)),
                ('bn4', nn.BatchNorm1d(output_size)),
                ('relu4', activ),
                ('avgpool', nn.AdaptiveAvgPool1d(1)),
            ]))
        else:
            self.feature_extractor = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels, 32, 5, padding=2)),
                ('relu1', activ),
                ('conv2', nn.Conv1d(32, 32, 3, padding=1)),
                ('relu2', activ),
                ('maxpool1', nn.MaxPool1d(2, 2)),
                ('conv3', nn.Conv1d(32, 64, 3, padding=1)),
                ('relu3', activ),
                ('maxpool2', nn.MaxPool1d(2, 2)),
                ('conv4', nn.Conv1d(64, output_size, 3, padding=1)),
                ('relu4', activ),
                ('avgpool', nn.AdaptiveAvgPool1d(1)),
            ]))

    def forward(self, x):
        features = self.feature_extractor(x).view(-1, self.output_size)
        return features


class CNN1D(nn.Module):
    '''
    CNN for time series classification
    '''

    def __init__(self, output_size, in_channels, batch_norm=False, dropout_prob=0.0):
        super().__init__()
        self.output_size = output_size
        self.in_channels = in_channels
        self.dropout_prob = dropout_prob

        activ = nn.ReLU(True)

        self.feature_extractor = CNN1DFeatureExtractor(in_channels, output_size=128,
                                                       batch_norm=False)
        # for p in self.parameters():
        #     p.requires_grad_ = False

        if batch_norm:
            self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(128, 1024)),
                ('bn1', nn.BatchNorm1D(1024)),
                ('relu1', activ),
                ('fc2', nn.Linear(1024, output_size)),
            ]))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(128, 1024)),
                ('relu1', activ),
                ('fc2', nn.Linear(1024, output_size)),
            ]))

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.dropout_prob > 0.0:
            features = nn.Dropout(self.dropout_prob)(features)
        logits = self.classifier(features)
        return logits


class CNN1DMultitask(MultitaskModel):
    '''
    CNN1D with Multitask heads
    '''
    def __init__(self, in_channels, task_dims, batch_norm=False, use_idx=None, dropout_prob=0.0):
        '''
        Constructor.

        Parameters
        ----------
        shared_dims : List[int]
            Defines the number and sizes of hidden layers that are shared
            amongst the tasks.
        task_dims : List[List[int]]
            Defines the number and sizes of hidden layers for a variable number
            of tasks.
        '''
        feature_size = 128
        shared_layers = CNN1DFeatureExtractor(in_channels, output_size=feature_size, batch_norm=batch_norm)
        super().__init__(feature_size, task_dims, shared_layers, batch_norm=batch_norm, use_idx=use_idx, dropout_prob=dropout_prob)
