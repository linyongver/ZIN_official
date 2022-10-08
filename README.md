# ZIN: Learning Invariance with Additional Auxiliary Information
The repo for ZIN: When and How to Learn Invariance by Environment Inference?

Our implementation about landcover is based on the source code of [In-N-Out](https://github.com/p-lambda/in-n-out)

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

