import os
import copy
import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

class FakeDataset(object):
    def __init__(self, n_groups, group_counts):
        self.n_groups = n_groups
        self._group_counts = group_counts

    def group_counts(self):
        return self._group_counts

    def group_str(self, fake_str):
        return ""


def align_envs(attrs_df, target_name, confounder_names, train_num, test_num, cons_train, cons_test):
    dfatr = attrs_df

    test_envs = len(cons_test)
    train_envs = len(cons_train)
    cons_ratio = cons_train + cons_test

    gvalues = [(-1, -1), (-1, 1), (1, 1), (1, -1)]

    target = target_name
    confunder = confounder_names #"Smiling" # "Mouth_Slightly_Open"
    dfall = dfatr.groupby(
        [target, confunder])["image_id"].count().reset_index()
    dfall["consistent"] = dfall[target] == dfall[confunder]
    dfall["image_id"].sum()

    train_env_nums = [int( train_num * 1. / train_envs)] * train_envs
    test_env_nums = [int( test_num * 1. / test_envs)] * test_envs

    avs = []
    for v1, v2 in gvalues:
        avs.append(dfall[(dfall[target] == v1) &
            (dfall[confunder] == v2)]["image_id"].iloc[0])
    all_cons = avs[0] + avs[2]
    all_incons = avs[1] + avs[3]

    train_env_con_nums = [cons * env_nums for (cons, env_nums)
        in zip(cons_ratio[: len(train_env_nums)], train_env_nums)]
    train_env_incon_nums = [
        env_nums - env_cons
        for (env_nums, env_cons)
        in zip(train_env_nums, train_env_con_nums)]
    test_env_con_nums = [cons * env_nums for (cons, env_nums)
        in zip(cons_ratio[len(train_env_nums):], test_env_nums)]
    test_env_incon_nums = [
        env_nums - env_cons
        for (env_nums, env_cons)
        in zip(test_env_nums, test_env_con_nums)]

    train_group_nums = []
    for cons, incons in zip(train_env_con_nums, train_env_incon_nums):
        train_group_nums.append(
            ( int(cons * avs[0] * 1. /  all_cons),
              int(incons * avs[1] * 1. /  all_incons),
              int(cons * avs[2] * 1. /  all_cons),
              int(incons * avs[3] * 1. /  all_incons)))
    # env3
    test_group_nums = []
    for cons, incons in zip(test_env_con_nums, test_env_incon_nums):
        test_group_nums.append(
            (int(cons * avs[0] * 1. /  all_cons),
              int(incons * avs[1] * 1. /  all_incons),
              int(cons * avs[2] * 1. /  all_cons),
              int(incons * avs[3] * 1. /  all_incons)))

    glists = []
    for gv in gvalues:
        g_list = dfatr[(dfatr[target] == gv[0]) &(dfatr[confunder] == gv[1])].reset_index()["index"].tolist()
        np.random.shuffle(g_list)
        glists.append(g_list)
    dfatr["env"] = -1
    envs = train_group_nums + test_group_nums
    for i in range(len(glists)):
        g_list = glists[i]
        loc = 0
        for e in range(len(envs)):
            dfatr["env"].iloc[g_list[loc: loc + envs[e][i]]] = e
            loc += envs[e][i]
    dfatr = dfatr[dfatr.env >= 0].reset_index()
    dff = dfatr[dfatr.env >= 0].groupby(["env", target, confunder])["image_id"].count().reset_index()
    for i in range(dff.shape[0]):
        row = dff.iloc[i]
        print("Env%s, %s=%s, %s=%s, Number=%s" % (
            row["env"], target_name, row[target_name], confounder_names, row[confounder_names], row["image_id"]))
    return dfatr

def get_transform_celebA(train, augment_data):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class SpuriousValDataset(Dataset):
    def __init__(self, val_dataset, train_envs):
        self.val_dataset = val_dataset
        self.train_envs = train_envs

    def __len__(self):
        return len(self.val_dataset)

    def __getitem__(self, idx):
        x,y,z,g,sp = self.val_dataset.__getitem__(idx)
        g = g - self.train_envs
        return x,y,z,g,sp

class SpuriousDataset(Dataset):
    def __init__(self, root_dir, target_name, confounder_names, auxilary_names, train_num, test_num, cons_train, cons_test):
        self.root_dir = root_dir
        self.train_num = train_num
        self.test_num = test_num
        self.cons_train = cons_train
        self.cons_test = cons_test
        self.target_name = target_name
        self.auxilary_names = auxilary_names
        self.confounder_names = confounder_names
        self.attrs_df = pd.read_csv(
            os.path.join(root_dir, 'data', 'list_attr_celeba.csv'))
        self.attrs_df = align_envs(self.attrs_df, target_name, confounder_names, train_num, test_num, cons_train, cons_test)
        self.data_dir = os.path.join(self.root_dir, 'data', 'img_align_celeba')
        self.filename_array = self.attrs_df['image_id'].values
        self.attr_names = self.attrs_df.columns.copy()
        self.y_array = self.attrs_df[self.target_name].values
        self.sp_array = (self.attrs_df[self.target_name].values == self.attrs_df[confounder_names].values).astype(np.int64)
        self.y_array[self.y_array == -1] = 0
        self.env_array = self.attrs_df["env"].values
        self.aux_array = self.attrs_df[self.auxilary_names].values
        self.split_array = self.attrs_df["env"].values
        # self.attrs_df = self.attrs_df.values
        # self.attrs_df[self.attrs_df == -1] = 0
        # target_idx = self.attr_idx(self.target_name)
        self.n_train_envs = len(self.cons_train)
        self.split_dict = {
            "train": list(range(len(self.cons_train))),
            "val": list(range(self.n_train_envs, self.n_train_envs + len(self.cons_test))),
            "test": list(range(self.n_train_envs, self.n_train_envs + len(self.cons_test)))}
        self.n_classes = 2
        self.train_transform = get_transform_celebA(train=True, augment_data=True)
        self.eval_transform = get_transform_celebA(train=False, augment_data=False)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.env_array[idx]
        z = self.aux_array[idx]
        sp = self.sp_array[idx]

        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        # Figure out split and transform accordingly
        if self.split_array[idx] in self.split_dict['train']:
            img = self.train_transform(img)
        elif self.split_array[idx] in self.split_dict['val'] + self.split_dict['test']:
            img = self.eval_transform(img)
        x = img

        return x,y,z,g,sp

    def get_splits(self, splits, train_frac=1.0):
        subsets = []
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            print(self.split_array[:20], self.split_dict[split])
            mask = np.isin(self.split_array, self.split_dict[split])
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if split == "train":
                subsets.append(Subset(self, indices))
            else:
                subsets.append(
                    SpuriousValDataset(
                        Subset(self, indices),
                        len(self.cons_train))
                )

        self.subsets = subsets
        return tuple(subsets)

    def get_tvt_fake_dataset_sp(self):
        raise Exception
        # train_fake_dataset = FakeDataset(n_groups=len(self.cons_ratio) - 1, group_counts=torch.Tensor([len(self.subsets[0]) / 2, len(self.subsets[0]) / 2]))
        # val_fake_dataset = FakeDataset(n_groups=1, group_counts=torch.Tensor([len(self.subsets[1])]))
        # test_fake_dataset = FakeDataset(n_groups=1, group_counts=torch.Tensor([len(self.subsets[2])]))
        # return train_fake_dataset, val_fake_dataset, test_fake_dataset


def get_data_loader_sp(root_dir, target_name, confounder_names, auxilary_names, batch_size, train_num, test_num, cons_train, cons_test):
    spd = SpuriousDataset(
        root_dir=root_dir,
        target_name=target_name,
        confounder_names=confounder_names,
        auxilary_names=auxilary_names,
        train_num=train_num,
        test_num=test_num,
        cons_train=cons_train,
        cons_test=cons_test)
    train_dataset, val_dataset, test_dataset = spd.get_splits(
        splits=['train','val','test'])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    train_data, val_data, test_data = train_dataset, val_dataset, test_dataset
    return spd, train_loader, val_loader, test_loader, train_data, val_data, test_data

if __name__ == "__main__":
    spd, train_loader, val_loader, test_loader, _, _, _ = get_data_loader_sp(
	root_dir="/home/jzhangey/datasets/Spurious/data/celeba",
	target_name="Smiling",
	confounder_names="Male",
        auxilary_names=["Young", "Blond_Hair", "Eyeglasses", "High_Cheekbones", "Big_Nose", "Bags_Under_Eyes", "Chubby"],
	batch_size=100,
	train_num=40000,
	test_num=20000,
	cons_train=[0.99, 0.9],
        cons_test=[0.001, 0.2, 0.8, 0.999])
    print(len(train_loader), len(val_loader), len(test_loader))
    x,y,z,g,sp = iter(copy.deepcopy(train_loader)).__next__()
    print(x.shape, y.shape, z.shape, g.shape, sp.shape)
    print(sp.float().mean())

    x,y,z,g,sp = iter(val_loader).__next__()
    print(x.shape, y.shape, z.shape, g.shape, sp.shape)
    print(sp.float().mean())
