import os
import sys
import subprocess
import socket
from argparse import ArgumentParser, REMAINDER

def parse_args():
    """
    Helper function to parse command-line arguments
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch helper "
                                        "that will spawn up multiple distributed processes")

    # Optional arguments
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set to "
                             "the number of GPUs in your system so that each process "
                             "can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0. For "
                             "single node multi-process training, the --master_addr "
                             "can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed training")
    parser.add_argument("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass 'local rank'. "
                             "If set to True, the script will not pass --local_rank "
                             "as an argument, and will instead set LOCAL_RANK.")
    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as "
                             "'python -m'.")

    # Positional arguments
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the training script")
    parser.add_argument('training_script_args', nargs=REMAINDER,
                        help="Additional arguments for the training script")

    return parser.parse_args()

def _find_free_port():
    """
    Find a free port
    @return int Free port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    args = parse_args()

    if "CUDA_VISIBLE_DEVICES" in os.environ and args.nproc_per_node == 1:
        args.nproc_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    # Calculate world size
    dist_world_size = args.nproc_per_node * args.nnodes
    port = args.master_port if args.master_addr != "127.0.0.1" else _find_free_port()

    # Set PyTorch distributed-related environment variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)

    for local_rank in range(0, args.nproc_per_node):
        # Each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # Build the command to launch the process
        cmd = [sys.executable, "-u"]

        if args.module:
            cmd.append("-m")

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        # Launch the process
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

if __name__ == "__main__":
    main()
