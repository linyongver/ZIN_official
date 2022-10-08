import torch 
from torch import optim, nn, autograd

class EIIL:
    def __init__(self, flags, dp):
        # only full gradient
        self.eiil_w = torch.randn([dp.data_num_train, 1]).cuda().requires_grad_()
        self.optimizer_eiil_w = optim.Adam([self.eiil_w], lr=0.001)
        self.flags = flags


    def __call__(self, batch_data, step,  mlp=None, scale=None, mean_nll=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = nn.MSELoss(reduction='none')
            train_nll = loss_fun(train_logits, train_y)
        elif self.flags.dataset == "landcover":
            train_nll = mean_nll(train_logits, train_y, reduction='none') # train_nll.shape=[bs]
        else:
            train_nll = nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")

        infered_envs = self.eiil_w.sigmoid() # shape=[bs, 1]
        env1_loss = (train_nll * infered_envs).mean()
        env2_loss = (train_nll * (1 - infered_envs)).mean()
        grad1 = autograd.grad(env1_loss, [scale], create_graph=True)[0]
        grad2 = autograd.grad(env2_loss, [scale], create_graph=True)[0]
        train_penalty = grad1 ** 2 + grad2 ** 2
        train_nll = train_nll.mean()

        self.optimizer_eiil_w.zero_grad()
        (-train_penalty).backward(retain_graph=True)
        self.optimizer_eiil_w.step()
        return train_nll, train_penalty

