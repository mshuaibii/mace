#!/bin/bash
# Parameters
#SBATCH --account=ocp
#SBATCH --cpus-per-task=8
#SBATCH --error=/fsx-checkpoints/mshuaibi/mace/%j_0_log.err
#SBATCH --output=/fsx-checkpoints/mshuaibi/mace/%j_0_log.out
#SBATCH --job-name=mace
#SBATCH --mem=80GB
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --qos=ocp_high
#SBATCH --time=10080

#train_set="/fsx-ocp-med/shared/omol_sample/train"
#val_set="/fsx-ocp-med/shared/omol_sample/val"
#train_set="/fsx-ocp-med/shared/omol_sample/ani2x/train"
#val_set="/fsx-ocp-med/shared/omol_sample/ani2x/val"
train_set="/fsx-ocp-med/shared/omol_sample/geom_orca6/train"
val_set="/fsx-ocp-med/shared/omol_sample/geom_orca6/val"

job_name="MACE-omol-geom-L2-3layers-float32"
srun /opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/mshuaibi/.local/share/mamba/envs/mace/bin/python mace/cli/run_train.py \
	--name=$job_name \
	--train_file=$train_set \
	--valid_file=$val_set \
	--statistics_file='/fsx-ocp-med/shared/omol_sample/omol_stats.json' \
	--energy_weight=40 \
	--forces_weight=1000 \
	--energy_key='energy' \
	--forces_key='forces' \
	--eval_interval=1 \
	--error_table='PerAtomMAE' \
	--model="ScaleShiftMACE" \
	--interaction_first="RealAgnosticResidualInteractionBlock" \
	--interaction="RealAgnosticResidualInteractionBlock" \
	--num_interactions=3 \
	--correlation=3 \
	--max_ell=3 \
	--r_max=6.0 \
	--max_L=2 \
	--num_channels=128 \
	--num_radial_basis=10 \
	--MLP_irreps="16x0e" \
	--scaling='rms_forces_scaling' \
	--mean=0 \
	--std=0.98 \
	--lr=0.01 \
	--weight_decay=1e-8 \
	--ema \
	--ema_decay=0.999 \
	--batch_size=16 \
	--valid_batch_size=16 \
	--max_num_epochs=200 \
	--patience=40 \
	--amsgrad \
	--device=cuda \
	--seed=1 \
	--clip_grad=1 \
	--keep_checkpoints \
	--save_all_checkpoints \
	--restart_latest \
	--default_dtype="float32" \
	--num_workers=4 \
	--save_cpu \
	--distributed \
	--wandb \
	--wandb_project="omol" \
	--wandb_entity="fairchem" \
	--wandb_name=$job_name
