import os
import torch
import importlib
import abc
import inspect
import torch.nn as nn
from argparse import Namespace


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITY = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x)
}


def load_model(path, config=False):  # **kwargs):  # we currently do not load the kwargs we may store
    data = torch.load(path)

    mod = importlib.import_module(data['module'])
    cls = getattr(mod, data['class_name'])
    model_config = data['model_config']
    ins = cls(**model_config)
    ins.load_state_dict(data['model_state_dict'], strict=False)
    if config:
        return ins, model_config
    return ins


def save_model(model, model_config, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        'module': model.__module__,
        'class_name': model.__class__.__name__,
        'model_config': model_config,
        'model_state_dict': model.state_dict()
    }
    data.update(kwargs)
    torch.save(data, path)


def delete_model(path):
    if os.path.exists(path):
        os.remove(path)


def init_model(module, class_name, config_dict, state_dict=None):
    mod = importlib.import_module(module)
    cls = getattr(mod, class_name)
    ins = cls(**config_dict)
    if state_dict:
        ins.load_state_dict(state_dict, strict=False)
    return ins
