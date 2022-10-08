from .model import InferEnv
import torch
import pandas as pd

class InferIrmV1:
    
    def __init__(self, flags, dp):
        if flags.dataset == "logit_z":
            infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset == "celebaz_feature":
            infer_env = InferEnv(flags, z_dim=7).cuda()
        elif flags.dataset == "house_price":
            infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset == "logit_2z":
            infer_env = InferEnv(flags, z_dim=flags.z_dim).cuda()
        self.flags = flags
        self.infer_env = infer_env
        self.optimizer_infer_env = torch.optim.Adam(infer_env.parameters(), lr=0.001) 
    
    def __call__(self, batch_data, step,  mlp=None, scale=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        normed_z = (train_z.float() - train_z.float().mean())/train_z.float().std()
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = torch.nn.MSELoss(reduction='none')
            train_nll = loss_fun(train_logits, train_y)
        else:
            train_nll = torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")
        infered_envs = self.infer_env(normed_z)
        env1_loss = (train_nll * infered_envs).mean()
        env2_loss = (train_nll * (1 - infered_envs)).mean()
        grad1 = torch.autograd.grad(
            env1_loss,
            [scale],
            create_graph=True)[0]
        grad2 = torch.autograd.grad(
            env2_loss,
            [scale],
            create_graph=True)[0]
        train_penalty = grad1 ** 2 + grad2 ** 2
        train_nll = train_nll.mean()

        if step < self.flags.penalty_anneal_iters:
            # gradient ascend on infer_env net
            self.optimizer_infer_env.zero_grad()
            (-train_penalty).backward(retain_graph=True)
            self.optimizer_infer_env.step()

        return train_nll, train_penalty