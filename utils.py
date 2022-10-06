import argparse
import math
import numpy as np
import torch
import os
import sys
from torch import nn, optim, autograd


def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def torch_xor(a, b):
    return (a-b).abs()

def concat_envs(con_envs):
    con_x = torch.cat([env["images"] for env in con_envs])
    con_y = torch.cat([env["labels"] for env in con_envs])
    con_g = torch.cat([
        ig * torch.ones_like(env["labels"])
        for ig,env in enumerate(con_envs)])
    # con_2g = torch.cat([
    #     (ig < (len(con_envs) // 2)) * torch.ones_like(env["labels"])
    #     for ig,env in enumerate(con_envs)]).long()
    con_c = torch.cat([env["color"] for env in con_envs])
    # con_yn = torch.cat([env["noise"] for env in con_envs])
    # return con_x, con_y, con_g, con_c
    return con_x.cuda(), con_y.cuda(), con_g.cuda(), con_c.cuda()


def merge_env(original_env, merged_num):
    merged_envs = merged_num
    a = original_env
    interval = (a.max() - a.min()) // merged_envs + 1
    b = (a - a.min()) // interval
    return b

def eval_acc_class(logits, labels, colors):
    acc  = mean_accuracy_class(logits, labels)
    minacc = mean_accuracy_class(
      logits[colors!=1],
      labels[colors!=1])
    majacc = mean_accuracy_class(
      logits[colors==1],
      labels[colors==1])
    return acc, minacc, majacc

def mean_accuracy_class(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def eval_acc_multi_class(logits, labels, colors):
    acc  = mean_accuracy_multi_class(logits, labels)
    minacc = mean_accuracy_multi_class(
      logits[colors.view(-1)!=1],
      labels[colors.view(-1)!=1])
    majacc = mean_accuracy_multi_class(
      logits[colors.view(-1)==1],
      labels[colors.view(-1)==1])
    return acc, minacc, majacc

def mean_accuracy_multi_class(output, target):
    probs = torch.softmax(output, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == target.view(-1))
    accuracy = corrects.sum().float() / float(corrects.size(0))
    return accuracy

def eval_acc_reg(logits, labels, colors):
    acc  = mean_nll_reg(logits, labels)
    minacc = torch.tensor(0.0)
    majacc = torch.tensor(0.0)
    return acc, minacc, majacc


def get_strctured_penalty(strctnet, ebd, envs_num, xis):
    x0, x1, x2 = xis
    assert envs_num > 2
    x2_ebd = ebd(x2).view(-1, 1) - 1
    x1_ebd = ebd(x1).view(-1, 1) - 1
    x0_ebd = ebd(x0).view(-1, 1) - 1
    x01_ebd = (x0_ebd-x1_ebd)[:, None]
    x12_ebd = (x1_ebd-x2_ebd)[:, None]
    x12_ebd_logit = strctnet(x01_ebd)
    return 10**13 * (x12_ebd_logit - x12_ebd).pow(2).mean()


def make_environment(images, labels, e):
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    color_mask = torch_bernoulli(e, len(labels))
    colors = torch_xor(labels, color_mask)
    # colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None],
      'color': (1- color_mask[:, None])
    }

def make_one_logit(num, sp_ratio, dim_inv, dim_spu):
    cc = CowCamels(
        dim_inv=dim_inv, dim_spu=dim_spu, n_envs=1,
        p=[sp_ratio], s= [0.5])
    inputs, outputs, colors, inv_noise= cc.sample(
        n=num, env="E0")
    return {
        'images': inputs,
        'labels': outputs,
        'color': colors
    }

def make_one_reg(num, sp_cond, inv_cond, dim_inv, dim_spu):
    ar = AntiReg(
        dim_inv=dim_inv, dim_spu=dim_spu, n_envs=1,
        s=[sp_cond], inv= [inv_cond])
    inputs, outputs, colors, inv_noise= ar.sample(
        n=num, env="E0")
    return {
        'images': inputs,
        'labels': outputs,
        'color': colors,
        'noise': None,
    }

def make_logit_envs(total_num, flags):
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef)/(envs_num-1) * i + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "cos":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.cos(i * 2.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "sin":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.sin(i * 2.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2cos":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.cos(i * 4.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2sin":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.sin(i * 4.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2_group":
        lower_coef = 0.8
        upper_coef = 0.9
        num_per_env = total_num // envs_num
        env_per_group = envs_num // 2
        for i in range(envs_num):
            env_coef = lower_coef if i <= env_per_group else upper_coef
            envs.append(
                make_one_logit(
                    num_per_env,
                    env_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    else:
        raise Exception
    envs.append(make_one_logit(total_num, 0.1, flags.dim_inv, flags.dim_spu))
    return envs

def make_reg_envs(total_num, flags):
    envs_num = flags.envs_num
    envs = []
    sp_ratio_list = [float(x) for x in flags.cons_ratio.split("_")]
    if flags.env_type == "linear":
        upper_coef = sp_ratio_list[0]
        lower_coef = sp_ratio_list[1]
        inv_cond = 1.0
        for i in range(envs_num):
            envs.append(
                make_one_reg(
                    total_num // envs_num,
                    (upper_coef - lower_coef)/(envs_num-1) * i + lower_coef,
                    inv_cond,
                    flags.dim_inv,
                    flags.dim_spu))
    else:
        raise Exception
    envs.append(make_one_reg(total_num, sp_ratio_list[-1], inv_cond, flags.dim_inv, flags.dim_spu))
    return envs


def mean_nll_class(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_nll_multi_class(logits, y):
    nll = nn.CrossEntropyLoss()
    return nll(logits, y.view(-1).long())

def mean_nll_reg(logits, y):
    l2loss = nn.MSELoss()
    return l2loss(logits, y)

def mean_accuracy_reg(logits, y, colors=None):
    return mean_nll_reg(logits, y)


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


class LYDataProvider(object):
    def __init__(self):
        pass

    def preprocess_data(self):
        pass

    def fetch_train(self):
        pass

    def fetch_test(self):
        pass

class LYDataProviderMK(LYDataProvider):
    def __init__(self, flags):
        super(LYDataProviderMK, self).__init__()

    def preprocess_data(self):
        self.train_x, self.train_y, self.train_g, self.train_c= concat_envs(self.envs[:-1])
        self.test_x, self.test_y, self.test_g, self.test_c= concat_envs(self.envs[-1:])

    def fetch_train(self):
        return self.train_x, self.train_y, self.train_g, self.train_c

    def fetch_test(self):
        return self.test_x, self.test_y, self.test_g, self.test_c

class CMNIST_LYDP(LYDataProviderMK):
    def __init__(self, flags):
        super(CMNIST_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_mnist_envs(flags)
        self.preprocess_data()
class LOGIT_LYDP(LYDataProviderMK):
    def __init__(self, flags):
        super(LOGIT_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_logit_envs(flags.data_num, flags)
        self.preprocess_data()

