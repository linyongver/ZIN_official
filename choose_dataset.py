from utils_z import CELEBAZ_FEATURE, HousePrice, LOGITZ, LOGIT2Z, LANDCOVER
from utils import LOGIT_LYDP, mean_nll_multi_class,eval_acc_multi_class,mean_accuracy_multi_class
from utils import eval_acc_class,eval_acc_reg,mean_nll_class,mean_accuracy_class,mean_nll_reg,mean_accuracy_reg

from model import MLP2Layer
from torch import nn 

def init_celebaz_feature(flags):
    dp = CELEBAZ_FEATURE(flags)
    feature_dim = dp.feature_dim
    hidden_dim = flags.hidden_dim
    mlp =  MLP2Layer(flags, feature_dim, hidden_dim).cuda()
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return dp, mlp, test_batch_num, train_batch_num, val_batch_num, test_batch_fetcher, mean_nll, mean_accuracy, eval_acc

def init_logit(flags):
    dp = LOGIT_LYDP(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mlp = nn.Linear(in_features=flags.dim_spu + flags.dim_inv, out_features=1).cuda()
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return dp, mlp, test_batch_num, train_batch_num, val_batch_num,test_batch_fetcher, mean_nll, mean_accuracy, eval_acc

def init_house_price(flags):
    dp = HousePrice(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    feature_dim = dp.feature_dim
    hidden_dim = flags.hidden_dim
    mlp =  MLP2Layer(flags, feature_dim, hidden_dim).cuda()
    mean_nll = mean_nll_reg
    mean_accuracy = mean_accuracy_reg
    eval_acc = eval_acc_reg    
    return dp, mlp, test_batch_num, train_batch_num, val_batch_num,test_batch_fetcher, mean_nll, mean_accuracy, eval_acc

def init_logit_z(flags):
    dp = LOGITZ(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu, out_features=1).cuda()
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return dp, mlp, test_batch_num, train_batch_num, val_batch_num,test_batch_fetcher, mean_nll, mean_accuracy, eval_acc

def init_logit_2z(flags):
    dp = LOGIT2Z(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu, out_features=1).cuda()
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return dp, mlp, test_batch_num, train_batch_num, val_batch_num,test_batch_fetcher, mean_nll, mean_accuracy, eval_acc


def init_landcover(flags):
    dp = LANDCOVER(flags)
    mlp = dp.fetch_mlp().cuda()
    test_batch_num = len(dp.test_loader)
    train_batch_num = len(dp.train_loader)
    val_batch_num = len(dp.val_loader)
    test_batch_fetcher = dp.fetch_test
    mean_nll = mean_nll_multi_class  # CrossEntropyLoss
    mean_accuracy = mean_accuracy_multi_class # test acc
    eval_acc = eval_acc_multi_class # train acc
    return dp, mlp, test_batch_num, train_batch_num, val_batch_num,test_batch_fetcher, mean_nll, mean_accuracy, eval_acc


def init_dataset(flags):
    dataset_specific_action = {
        "celebaz_feature": init_celebaz_feature,
        "house_price": init_house_price,
        "logit": init_logit,
        "logit_z": init_logit_z,
        "logit_2z": init_logit_2z,
        "landcover": init_landcover
    }
    return dataset_specific_action[flags.dataset](flags)