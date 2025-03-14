###########################################################################################
# Slurm environment setup for distributed training.
# This code is refactored from rsarm's contribution at:
# https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import os
import torch
import logging

import hostlist
import subprocess
import torch.distributed as dist
from datetime import timedelta

DISTRIBUTED_PORT = 13356
CURRENT_DEVICE_STR = "CURRRENT_DEVICE"


class DistributedEnvironment:
    def __init__(self):
        self._setup_distr_env()
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])

    def _setup_distr_env(self):
        hostname = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0]
        os.environ["MASTER_ADDR"] = hostname
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "33333")
        os.environ["WORLD_SIZE"] = os.environ.get(
            "SLURM_NTASKS",
            str(
                int(os.environ["SLURM_NTASKS_PER_NODE"])
                * int(os.environ["SLURM_NNODES"])
            ),
        )
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]


def setup():
    config = {
        "world_size": int(os.environ["SLURM_NTASKS_PER_NODE"]) * int(os.environ["SLURM_NNODES"]),
    }
    timeout = timedelta(minutes=config.get("timeout", 30))
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list is not None:
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", node_list]
        )
        config["init_method"] = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=DISTRIBUTED_PORT,
        )
        nnodes = int(os_environ_get_or_throw("SLURM_NNODES"))
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
        else:
            ntasks = int(os_environ_get_or_throw("SLURM_NTASKS"))
            nnodes = int(os_environ_get_or_throw("SLURM_NNODES"))
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)
        if ntasks_per_node == 1:
            assert config["world_size"] % nnodes == 0
            gpus_per_node = config["world_size"] // nnodes
            node_id = int(os_environ_get_or_throw("SLURM_NODEID"))
            config["rank"] = node_id * gpus_per_node
            config["local_rank"] = 0
        else:
            assert ntasks_per_node == config["world_size"] // nnodes
            config["rank"] = int(os_environ_get_or_throw("SLURM_PROCID"))
            config["local_rank"] = int(os_environ_get_or_throw("SLURM_LOCALID"))

        logging.info(
            f"Init: {config['init_method']}, {config['world_size']}, {config['rank']}"
        )

        # ensures GPU0 does not have extra context/higher peak memory
        logging.info(
            f"local rank: {config['local_rank']}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        # in the old code, all ranks can see all devices but need to be assigned a device equal to their local rank
        # this is dangerous and should be deprecated, however, FSDP still requires backwards compatibility with
        # initializing this way for now so we need to keep it
        torch.cuda.set_device(config["local_rank"])

        dist.init_process_group(
            backend="nccl",
            init_method=config["init_method"],
            world_size=config["world_size"],
            rank=config["rank"],
            timeout=timeout,
        )
        return config


def assign_device_for_local_rank(cpu: bool, local_rank: int):
    if cpu:
        os.environ[CURRENT_DEVICE_STR] = "cpu"
    else:
        # assert the cuda device to be the local rank
        os.environ[CURRENT_DEVICE_STR] = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)


def get_device_for_local_rank():
    cur_dev_env = os.environ.get(CURRENT_DEVICE_STR)
    if cur_dev_env is not None:
        return cur_dev_env
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.warning(
            f"{CURRENT_DEVICE_STR} env variable not found, defaulting to {device}"
        )
        return device


def os_environ_get_or_throw(x: str) -> str:
    if x not in os.environ:
        raise RuntimeError(f"Could not find {x} in ENV variables")
    return os.environ.get(x)
