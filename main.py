import argparse
import numpy as np
import torch
from torch import optim

from utils_z import MetaAcc, pretty_print_ly
from utils import  CosineLR

from choose_dataset import init_dataset 
from algorithms import algorithm_builder

parser = argparse.ArgumentParser(description='ZIN')
parser.add_argument('--aux_num', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes_num', type=int, default=2)
parser.add_argument('--dataset', type=str, default="mnist", choices=["celebaz_feature", "logit", "logit_z", "logit_2z",  "house_price", "landcover"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=100)
parser.add_argument('--dim_inv', type=int, default=2)
parser.add_argument('--dim_spu', type=int, default=10)
parser.add_argument('--data_num_train', type=int, default=2000)
parser.add_argument('--data_num_test', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="irmv1", type=str, choices=["erm", "infer_irmv1", "infer_irmv1_multi_class", "lff", "eiil"])
parser.add_argument('--n_restarts', type=int, default=10)
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
parser.add_argument('--scheduler', type=int, default=0)
flags = parser.parse_args()
print("batch_size is", flags.batch_size)

torch.manual_seed(flags.seed)
np.random.seed(flags.seed+111)

flags.cons_ratio = "_".join([flags.cons_train, flags.cons_test])
flags.envs_num_train = len(flags.cons_train.split("_"))
flags.envs_num_test = len(flags.cons_test.split("_"))
assert flags.envs_num_test + flags.envs_num_train == len(flags.cons_ratio.split("_"))

for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    dp, mlp, test_batch_num, train_batch_num, val_batch_num,test_batch_fetcher, mean_nll, mean_accuracy, eval_acc = init_dataset(flags)
   
    if flags.opt == "adam":
        optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    elif flags.opt == "sgd":
        optimizer = optim.SGD(mlp.parameters(), momentum=0.9, lr=flags.lr)

    if flags.scheduler:
        scheduler = CosineLR(optimizer, flags.lr, flags.steps)

    scale = torch.tensor(1.).cuda().requires_grad_()
    algo = algorithm_builder(flags, dp)

    if flags.dataset == "house_price":
        meta_acc_test = MetaAcc(env=dp.envs_num_test, acc_measure=mean_accuracy, acc_type="test")
    else:
        meta_acc_test = MetaAcc(env=flags.envs_num_test, acc_measure=mean_accuracy, acc_type="test")
    
    pretty_print_ly(['step', 'train penalty', 'train acc'] + meta_acc_test.acc_fields)
    best_results = {'val_acc': 0.0, 'epoch': -1, 'train_acc': 0.0, 'test_acc': 0.0}
    for step in range(flags.steps):
        mlp.train()
        for batch in range(train_batch_num): 
            batch_data = dp.fetch_train()
            train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data

            # calculate train loss for different algorithms and datasets 
            train_nll, train_penalty = algo(batch_data, step, mlp, scale, mean_nll=mean_nll)

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            penalty_weight = (flags.penalty_weight
                if step >= flags.penalty_anneal_iters else 0.0)
            if flags.irm_type == "erm":
                penalty_weight = 0
            train_penalty = torch.max(torch.tensor(0.0).cuda(), train_penalty.cuda())
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= (1. + penalty_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if flags.scheduler:
            scheduler.step()

        if step % flags.print_every == 0:
            mlp.eval()

            train_accs = []
            for ii in range(train_batch_num):
                train_x, train_y, train_z, train_g, train_c, train_invnoise = dp.fetch_train()
                train_logits = mlp(train_x)
                train_acc, train_minacc, train_majacc = eval_acc(train_logits, train_y, train_c) 
                train_accs.append(train_acc)
            train_acc = sum(train_accs) / len(train_accs)

            meta_acc_test.clear()
            for ii in range(test_batch_num):
                test_x, test_y, test_z, test_g, test_c, test_invnoise = test_batch_fetcher()
                test_logits = mlp(test_x)
                meta_acc_test.process_batch(test_y, test_logits, test_g)
            meta_acc_test_res = meta_acc_test.meta_acc

            if val_batch_num:
                val_accs = []
                for ii in range(val_batch_num):
                    val_x, val_y, val_z, val_g, val_c, val_invnoise = dp.fetch_val()
                    val_logits = mlp(val_x)
                    val_acc, val_minacc, val_majacc = eval_acc(val_logits, val_y, val_c) 
                    val_accs.append(val_acc)
                val_acc = sum(val_accs) / len(val_accs)
                if val_acc > best_results['val_acc']:
                    best_results['val_acc'] = val_acc
                    best_results['epoch'] = step
                    best_results['train_acc'] = train_acc
                    best_results['test_acc'] = meta_acc_test_res['test_acc']
        
            pretty_print_ly(
                [np.int32(step),
                train_penalty.detach().cpu().numpy(),
                np.float64(train_acc)] + 
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
    
    if val_batch_num:
        print("best results:")
        print(best_results)



