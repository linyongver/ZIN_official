#!/bin/bash
# MCOLOR Experiment.

# Hyperparameters
N_RESTARTS=3
HIDDEN_DIM=390
L2_REGULARIZER_WEIGHT=0.00110794568
LR=0.0004898536566546834
LABEL_NOISE=${1-0.05}
PENALTY_ANNEAL_ITERS=190
PENALTY_WEIGHT=191257.18613115903
STEPS=501
ROOT=${results/mcolor}



# ERM
python -u -m opt_env.irm_mcolor \
  --results_dir results/mcolor/erm \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --penalty_anneal_iters 0 \
  --penalty_weight 0.0 \
  --steps $STEPS


# IRM
python -u -m opt_env.irm_mcolor \
  --results_dir results/mcolor/irm \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS


  
# EIIL
python -u -m opt_env.irm_mcolor \
  --results_dir results/mcolor/eiil \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS \
  --eiil
