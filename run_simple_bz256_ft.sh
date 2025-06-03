#!/bin/bash
#SBATCH --account=ocp
#SBATCH --cpus-per-task=8
#SBATCH --error=/checkpoint/ocp/mshuaibi/omol/mace/%j_0_log.err
#SBATCH --output=/checkpoint/ocp/mshuaibi/omol/mace/%j_0_log.out
#SBATCH --job-name=mace
#SBATCH --mem=1000GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --qos=ocp_high
#SBATCH --exclusive
#SBATCH --time=10080

#h100-2
#train_set="/opt/hpcaas/.mounts/fs-0a14d5cae11d2d8c0/shared/omol/250409-final/train"
#val_set="/opt/hpcaas/.mounts/fs-0a14d5cae11d2d8c0/shared/omol/250409-final/val_30k"
#scale_file="/opt/hpcaas/.mounts/fs-0a14d5cae11d2d8c0/mshuaibi/omol/mace/omol_stats_041025.json"
#h100-1
train_set="/opt/hpcaas/.mounts/fs-0c40489436c32db98/shared/omol/250409-final/simple_train"
val_set="/opt/hpcaas/.mounts/fs-0c40489436c32db98/shared/omol/250409-final/simple_val"
scale_file="/opt/hpcaas/.mounts/fs-0c40489436c32db98/shared/omol/250409-final/mace/omol_stats_041025.json"
work_dir="/opt/hpcaas/.mounts/fs-0c40489436c32db98/shared/omol/run_dir/mace"
PYTHON="/opt/hpcaas/.mounts/fs-072917c00f01ae1ba/home/mshuaibi/.local/share/mamba/envs/mace/bin/python"

job_name="042525_mace_bz256_simple_ft"
srun $PYTHON mace/cli/run_train.py \
    --name=$job_name \
    --train_file=$train_set \
    --valid_file=$val_set \
    --statistics_file=$scale_file \
    --energy_weight=100 \
    --forces_weight=10 \
    --energy_key='energy' \
    --forces_key='forces' \
    --eval_interval=1 \
    --eval_interval_steps=5000 \
    --error_table='PerAtomMAE' \
    --model="ScaleShiftMACE" \
    --loss='l1l2_forces' \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=3 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=7.0 \
    --max_L=1 \
    --num_channels=512 \
    --num_radial_basis=8 \
    --MLP_irreps="16x0e" \
    --scaling='rms_forces_scaling' \
    --mean=0 \
    --std=1.429279 \
    --lr=0.01 \
    --weight_decay=0.0 \
    --ema \
    --ema_decay=0.999 \
    --batch_size=16 \
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
    --save_cpu \
    --work_dir $work_dir \
    --distributed \
    --wandb \
    --wandb_project="omol" \
    --wandb_entity="fairchem" \
    --wandb_name=$job_name \
    --enable_cueq=True
