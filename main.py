import argparse
from utils_z import HousePrice
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import os
import sys
from torch import nn, optim, autograd
from model import ENV_EBD, PredYEnvHatY
from model import EBD
from model import BayesW
from model import Y_EBD, PredEnvHatY, PredEnvHatYSep, PredEnvYY
from model import InferEnv
from model import InferEnvMultiClass
from model import MLP2Layer

from utils import concat_envs,eval_acc_class,eval_acc_reg,mean_nll_class,mean_accuracy_class,mean_nll_reg,mean_accuracy_reg
from utils import mean_nll_multi_class,eval_acc_multi_class,mean_accuracy_multi_class
from utils_z import MetaAcc, pretty_print_ly
from utils_z import CELEBAZ_FEATURE
from utils_z import LOGITZ
from utils_z import LOGIT2Z

parser = argparse.ArgumentParser(description='ZIN')
parser.add_argument('--aux_num', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes_num', type=int, default=2)
parser.add_argument('--dataset', type=str, default="mnist", choices=["celebaz_feature", "house_price", "logit", "logit_z", "logit_2z"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=100)
parser.add_argument('--dim_inv', type=int, default=2)
parser.add_argument('--dim_spu', type=int, default=10)
parser.add_argument('--data_num_train', type=int, default=2000)
parser.add_argument('--data_num_test', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="infer_irmv1_multi_class", type=str, choices=["erm", "infer_irmv1", "infer_irmv1_multi_class"])
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--save_last', default=0, type=int, choices=[0, 1])
parser.add_argument('--image_scale', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--hidden_dim_infer', type=int, default=16)
parser.add_argument('--cons_train', type=str, default="0.999_0.7")
parser.add_argument('--cons_test', type=str, default="0.999_0.001")
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--z_class_num', type=int, default=4)
parser.add_argument('--noise_ratio', type=float, default=0.1)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
flags = parser.parse_args()
print("batch_size is", flags.batch_size)

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)

flags.cons_ratio = "_".join([flags.cons_train, flags.cons_test])
flags.envs_num_train = len(flags.cons_train.split("_"))
flags.envs_num_test = len(flags.cons_test.split("_"))
assert flags.envs_num_test + flags.envs_num_train == len(flags.cons_ratio.split("_"))
irm_type = flags.irm_type


for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))


tmp_out = []
final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)
    if flags.dataset == "celebaz_feature":
        dp = CELEBAZ_FEATURE(flags)
        feature_dim = dp.feature_dim
        hidden_dim = flags.hidden_dim
        mlp =  MLP2Layer(flags, feature_dim, hidden_dim).cuda()
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    elif flags.dataset == "logit":
        dp = LOGIT_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = nn.Linear(in_features=flags.dim_spu + flags.dim_inv, out_features=1).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    elif flags.dataset == "house_price":
        dp = HousePrice(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        feature_dim = dp.feature_dim
        hidden_dim = flags.hidden_dim
        mlp =  MLP2Layer(flags, feature_dim, hidden_dim).cuda()
        mean_nll = mean_nll_reg
        mean_accuracy = mean_accuracy_reg
        eval_acc = eval_acc_reg
    elif flags.dataset == "logit_z":
        dp = LOGITZ(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu, out_features=1).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    else:
        raise Exception
    if flags.opt == "adam":
        optimizer = optim.Adam(
          mlp.parameters(),
          lr=flags.lr)
    elif flags.opt == "sgd":
        optimizer = optim.SGD(
          mlp.parameters(),
          momentum=0.9,
          lr=flags.lr)
    else:
        raise Exception

    if flags.irm_type == "infer_irmv1":
        scale = torch.tensor(1.).cuda().requires_grad_()
        if flags.dataset == "logit_z":
            infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset == "celebaz_feature":
            infer_env = InferEnv(flags, z_dim=7).cuda()
        elif flags.dataset == "house_price":
            infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset in ["logit_2z", "logit_2z_ab_z1", "logit_2z_ab_z2", "logit_2z_ab_x", "logit_2z_ab_xy", "logit_2z_ab_zxy"]:
            infer_env = InferEnv(flags, z_dim=dp.z_dim).cuda()
        else:
            raise Exception
        optimizer_infer_env = optim.Adam(
          infer_env.parameters(),
          lr=0.001)
    elif flags.irm_type == "infer_irmv1_multi_class":
        scale = torch.tensor(1.).cuda().requires_grad_()
        if flags.dataset == "logit_z":
            infer_env = InferEnvMultiClass(flags, z_dim=1, class_num=flags.z_class_num).cuda()
        elif flags.dataset == "celebaz_feature":
            infer_env = InferEnvMultiClass(flags, z_dim=7, class_num=flags.z_class_num).cuda()
        elif flags.dataset == "house_price":
            infer_env = InferEnvMultiClass(flags, z_dim=1, class_num=flags.z_class_num).cuda()
        else:
            raise Exception
        optimizer_infer_env = optim.Adam(
          infer_env.parameters(),
          lr=0.001)
    else:
        pass

    if flags.dataset in ["house_price"]:
        meta_acc_test = MetaAcc(env=dp.envs_num_test, acc_measure=mean_accuracy, acc_type="test")
    else:
        meta_acc_test = MetaAcc(env=flags.envs_num_test, acc_measure=mean_accuracy, acc_type="test")
    pretty_print_ly(['step', 'train penalty'] + ["train_acc"] + meta_acc_test.acc_fields)
    for step in range(flags.steps):
        mlp.train()
        train_x, train_y, train_z, train_g, train_c, train_invnoise= dp.fetch_train()
        if irm_type == "erm":
            train_logits = mlp(train_x)
            train_nll = mean_nll(train_logits, train_y)
            train_penalty = torch.tensor(0.0)
        elif flags.irm_type == "infer_irmv1":
            normed_z = (train_z.float() - train_z.float().mean())/train_z.float().std()
            train_logits = scale * mlp(train_x)
            if flags.dataset == "house_price":
                loss_fun = nn.MSELoss(reduction='none')
                train_nll = loss_fun(train_logits, train_y)
            else:
                train_nll = nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")
            infered_envs = infer_env(normed_z)
            env1_loss = (train_nll * infered_envs).mean()
            env2_loss = (train_nll * (1 - infered_envs)).mean()
            grad1 = autograd.grad(
                env1_loss,
                [scale],
                create_graph=True)[0]
            grad2 = autograd.grad(
                env2_loss,
                [scale],
                create_graph=True)[0]
            train_penalty = grad1 ** 2 + grad2 ** 2
            train_nll = train_nll.mean()

            if step < flags.penalty_anneal_iters:
                # gradient ascend on infer_env net
                optimizer_infer_env.zero_grad()
                (-train_penalty).backward(retain_graph=True)
                optimizer_infer_env.step()
        elif flags.irm_type == "infer_irmv1_multi_class":
            normed_z = (train_z.float() - train_z.float().mean())/train_z.float().std()
            train_logits = scale * mlp(train_x)
            if flags.dataset == "house_price":
                loss_fun = nn.MSELoss(reduction='none')
                train_nll = loss_fun(train_logits, train_y)
            else:
                train_nll = nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")
            infered_envs = infer_env(normed_z)
            train_penalty = 0
            multi_loss = (train_nll * infered_envs).mean(axis=0) 
            for i in range(multi_loss.shape[0]):
                grad = autograd.grad(
                    multi_loss[i],
                    [scale],
                    create_graph=True)[0]
                train_penalty += grad ** 2
            train_nll = train_nll.mean()

            if step < flags.penalty_anneal_iters:
                # gradient ascend on infer_env net
                optimizer_infer_env.zero_grad()
                (-train_penalty).backward(retain_graph=True)
                optimizer_infer_env.step()
        else:
            raise Exception

        train_acc, train_minacc, train_majacc = eval_acc(train_logits, train_y, train_c)
        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
            if step >= flags.penalty_anneal_iters else 0.0)
        if flags.irm_type == "erm":
            penalty_weight = 0
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
          loss /= (1. + penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_schd.step()

        if step % flags.print_every == 0:
            mlp.eval()
            meta_acc_test.clear()
            for ii in range(test_batch_num):
                test_x, test_y, test_z, test_g, test_c, test_invnoise = test_batch_fetcher()
                test_logits = mlp(test_x)
                meta_acc_test.process_batch(test_y, test_logits, test_g)
            meta_acc_test_res = meta_acc_test.meta_acc
            pretty_print_ly(
                [np.int32(step),
                train_penalty.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy()] +
                [meta_acc_test_res[fd].detach().cpu().numpy() for fd in meta_acc_test.acc_fields])
            stats_dict = {
                "train_nll": train_nll.detach().cpu().numpy(),
                "train_acc": train_acc.detach().cpu().numpy(),
                "train_minacc": train_minacc.detach().cpu().numpy(),
                "train_majacc": train_majacc.detach().cpu().numpy(),
                "train_penalty": train_penalty.detach().cpu().numpy(),
            }
            stats_dict.update(
                dict(zip(
                    meta_acc_test.acc_fields,
                    [meta_acc_test_res[fd].detach().cpu().numpy() for fd in meta_acc_test.acc_fields]
                ))
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(meta_acc_test_res["test_acc"].detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
