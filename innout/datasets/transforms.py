import torch
from torchvision import transforms
import importlib


def to_tensor(np_array):
    '''
    Numpy to Tensor, but doesn't add unnecessary new dimensions

    Parameters
    ----------
    nd_array : numpy.ndarray
        NumPy array to convert.

    Returns
    -------
    torch.Tensor containing the data.
    '''
    return torch.from_numpy(np_array)


def flatten_to_tensor(np_array):
    '''
    Flattens a NumPy ndarray and converts it to a PyTorch Tensor object.

    Parameters
    ----------
    nd_array : numpy.ndarray
        NumPy array to convert.

    Returns
    -------
    torch.Tensor containing the flattened data.
    '''
    return torch.from_numpy(np_array.flatten())


def tensor_to_float(tensor):
    '''
    Wrapper around torch.Tensor.float().

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    torch.Tensor
    '''
    return tensor.float()


class LambdaTransform(object):
    '''
    Wrapper around torchvision.transforms.Lambda.
    '''
    def __init__(self, function_path, **kwargs):
        '''
        Constructs a transform that calls the function specified by
        function_path.

        Parameters
        ----------
        function_path : str
            Import path of the function, e.g., "numpy.mean".
        **kwargs
            Any static keyword arguments required for the function. For
            example, axis=0 could be given to numpy.mean.
        '''
        module_name, function_name = function_path.rsplit(".", 1)
        func = getattr(importlib.import_module(module_name), function_name)
        self.kwargs = kwargs
        self.transform = transforms.Lambda(lambda x: func(x, **self.kwargs))

    def __call__(self, sample):
        '''
        Calls the specified method on the sample.

        Parameters
        ----------
        sample : Any
            Whatever input self.transform is expecting.
        '''
        return self.transform(sample)
