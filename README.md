# ZIN: When and How to Learn Invariance Without Environment Partition?
The repo for ZIN: When and How to Learn Invariance Without Environment Partition?


# Requirements
## Environment
Our code works with the following environment:
* pytorch-transformers==1.2.0
* torch==1.3.1
* torchvision==0.4.2
* Pillow==6.1

To install the necessary packages for the project, please run: 
```
pip install -r requirements.txt
```


## Datasets
Run the following command to download the datasets: CelebA, house_price, landcover. They will be put in datasets directory.
```
bash data_downloader.sh
```
Dataset Discriptions:
* HousePrice.  This is implemented based on the house price dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
* CelebA. The dataset is from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Since IRM suffers from overfitting problem when applied to large models like ResNet-18, we fix the feature extraction backbone (that is, use a pre-trained ResNet-18 with fixed blocks) in this task. One may avoid this limitation by incorperate overfitting robust IRM variants like [1][2][3].
* Landcover. Our implementation on landcover is based on the source code of [In-N-Out](https://github.com/p-lambda/in-n-out). Notably, we use exact the same random seed as In-N-Out.

[1] Yong Lin, Hanze Dong, Hao Wang, Tong Zhang, Bayesian Invariant Risk Minimization, CVPR 2022 
[2] Xiao Zhou, Yong Lin, Weizhong Zhang, Tong Zhang, Sparse Invariant Risk Minimization, ICML 2022
[3] Xiao Zhou, Yong Lin, Renjie Pi, Weizhong Zhang, Renzhe Xu, Peng Cui, Tong Zhang., Model Agnostic Sample Reweighting for Out-of-Distribution Learning, ICML 2022

# Quick Start (For Reproducing Results)
1. To run the ZIN in the temporal dataset with setting p_s=(0.999, 0.9) and p_v=0.9.
    ```
    python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.1 --cons_train 0.999_0.9 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 10000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type infer_irmv1 --dataset logit_z --penalty_anneal_iters 5000
    ```

    The expected test accuracy is about `82.96`.

2. To run ZIN in CelebA dataset with 7 auxiliary information: "Young", "Blond_Hair", "Eyeglasses", "High_Cheekbones", "Big_Nose", "Bags_Under_Eyes", "Chubby"
    ```
    python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.2 --cons_train 0.999_0.8 --cons_test 0.01_0.2_0.8_0.999 --penalty_weight 10000 --steps 8500 --dim_inv 5 --dim_sp 5 --data_num_train 39996 --data_num_test 20000 --n_restarts 1 --irm_type infer_irmv1 --dataset celebaz_feature --penalty_anneal_iters 8000 --seed 1
    ```

    The expected test accuracy is about `76.29`.

3. To run ZIN in house price prediction task
    ```
    python main.py --l2_regularizer_weight 0.001 --lr 0.005 --penalty_weight 1000 --steps 3801   --n_restarts 1 --irm_type infer_irmv1_multi_class --dataset house_price --penalty_anneal_iters 3000 --seed 2 --hidden_dim_infer 64 --hidden_dim 32 
    ```


4. To run the ZIN with location(latitude and longitude) as auxiliary information in landcover dataset:
    ```
    python main.py --aux_num 2 --batch_size 1024 --seed 112 --classes_num 6 --dataset landcover --opt adam --l2_regularizer_weight 0.001 --print_every 1 --lr 0.1 --irm_type infer_irmv1_multi_class --n_restarts 1 --num_classes 6 --z_class_num 2 --penalty_anneal_iters 40 --penalty_weight 10 --steps 400 --scheduler 1
    ```
    The expected test accuracy is about `66.06`.

## Use ZIN on Your Own Data
We provider interface for you to include your own data. You need to inherit the 
 class `LYDataProviderMK`, and re-implement the function `fetch_train` and `fetch_test`. The main function will call `fetch_train` to get training data for each step. `fetch_train` should return the following values:

* `train_x`: the feature tensor;
* `train_y`: the label tensor;
* `train_z`: the auxilary information tensor;
* `train_g`(optional, can set to be `None`): the tensor contains values indicating which environmnets the data are from;
* `train_c`(optional, can set to be `None`): the tensor contains values indicating whether the spurious features align with the labels.
* `train_invnoise`(optional, can set to be `None`): the tensor contains values indicating the noisy ratio of the label;

The structure of the return value of `fetch_test` are similar with `fetch_train`.
# Contact Information

For help or issues using ZIN, please submit a GitHub issue.

For personal communication related to ZIN, please contact Yong Lin (`ylindf@connect.ust.hk`).

# Reference 
If you use or extend our work, please cite the following paper:
```
@inproceedings{lin2022zin,
  title={ZIN: When and How to Learn Invariance Without Environment Partition?},
  author={Lin, Yong and Zhu, Shengyu and Tan, Lu and Cui, Peng},
  booktitle={Advances in neural information processing systems},
  year={2022}
}
```
