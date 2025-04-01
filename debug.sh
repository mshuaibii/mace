#!/bin/bash
train_set="/fsx-ocp-med/shared/omol_sample/train"
val_set="/fsx-ocp-med/shared/omol_sample/val"
scale_file="/fsx-ocp-med/shared/omol_sample/omol_stats.json"

job_name="MACE-omol-L1-3layers-linear-l1l2-big-512-ok"
/opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/mshuaibi/.local/share/mamba/envs/mace/bin/python mace/cli/run_train.py \
    --name=$job_name \
    --train_file=$train_set \
    --valid_file=$val_set \
    --statistics_file=$scale_file \
    --energy_weight=10 \
    --forces_weight=10 \
    --energy_key='energy' \
    --forces_key='forces' \
    --eval_interval=1 \
    --error_table='PerAtomMAE' \
    --model="ScaleShiftMACE" \
    --loss='l1l2_forces' \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=3 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=6.0 \
    --max_L=1 \
    --num_channels=512 \
    --num_radial_basis=8 \
    --MLP_irreps="16x0e" \
    --scaling='rms_forces_scaling' \
    --mean=0 \
    --std=0.98 \
    --lr=0.01 \
    --weight_decay=0.0 \
    --ema \
    --ema_decay=0.999 \
    --batch_size=4 \
    --valid_batch_size=16 \
    --max_num_epochs=200 \
    --optimizer="schedulefree" \
    --patience=40 \
    --amsgrad \
    --device=cuda \
    --seed=1 \
    --clip_grad=100 \
    --keep_checkpoints \
    --save_all_checkpoints \
    --restart_latest \
    --default_dtype="float32" \
    --num_workers=4 \
    --save_cpu
