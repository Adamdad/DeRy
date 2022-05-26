import os
import subprocess
import time
import argparse
def assign_free_gpus(threshold_vram_usage=1500, max_gpus=1, wait=True, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        if len(free_gpus) < max_gpus:
            return False
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")

def assign_free_port():
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    free_port = sock.getsockname()[1]
    free_port = str(free_port)
    os.environ["PORT"] = free_port
    print(f"Using PORT: {free_port}")


parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('config', help='train config file path')
parser.add_argument('GPUS', type=int, help='number of gpus')
# parser.add_argument('ARGS', type=str, default='', help='number of gpus')
args = parser.parse_args()
assign_free_gpus(threshold_vram_usage=1500, max_gpus=args.GPUS, wait=True, sleep_time=3 * 60)
assign_free_port()
os.system(f'bash tools/dist_train.sh {args.config} {args.GPUS}') #{args.ARGS}