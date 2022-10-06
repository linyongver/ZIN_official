
# Requirements
* pytorch-transformers==1.2.0
* torch==1.3.1
* torchvision==0.4.2
* Pillow==6.1

# Quick Start
## Run Synthetic Dataset
Simple CMD to run the ZIN in the temporal dataset with setting p_s=(0.999, 0.9) and p_v=0.9.
```
python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.1 --cons_train 0.999_0.9 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 10000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type infer_irmv1 --dataset logit_z --penalty_anneal_iters 5000
```
## Run ClebeA
```
python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.2 --cons_train 0.999_0.8 --cons_test 0.01_0.2_0.8_0.999 --penalty_weight 10000 --steps 8500 --dim_inv 5 --dim_sp 5 --data_num_train 39996 --data_num_test 20000 --n_restarts 1 --irm_type infer_irmv1 --dataset celebaz_feature --penalty_anneal_iters 8000 --seed 1
```
### Run House Price
```
python main.py --l2_regularizer_weight 0.001 --lr 0.005 --penalty_weight 1000 --steps 3801   --n_restarts 1 --irm_type infer_irmv1_multi_class --dataset house_price --penalty_anneal_iters 3000 --seed 2 --hidden_dim_infer 64 --hidden_dim 32 
```
