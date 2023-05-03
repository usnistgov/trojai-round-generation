import os
import numpy as np
import torch
import multiprocessing
import subprocess
import logging


# due to pytorch + numpy bug
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def hex_to_rgb(value):
    if value == 'any':
        return value
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))


def is_ide_debug_mode():
    # check if IDE is in debug mode, and set num parallel worker to 0
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        logging.info("Detected IDE debug mode, setting number of workers to 0 to allow IDE debugger to work with pytorch.")
        return True
    return False


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
    return memory_used_percent, memory_total_info


def get_num_workers():
    # default to all the cores
    num_workers = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError as e:
        pass  # do nothing

    if is_ide_debug_mode():
        # IDE is debug (works at least of PyCharm), so set num workers to 0
        num_workers = 0
    # if config.DEBUGGING_FLAG:
    #    num_workers = 0
    logging.info("Using {} Workers".format(num_workers))
    return num_workers


def clamp(X, l, u, cuda=True):
    """
    Clamps a tensor to lower bound l and upper bound u.

    Args:
        X: the tensor to clamp.
        l: lower bound for the clamp.
        u: upper bound for the clamp.
        cuda: whether the tensor should be on the gpu.
    """

    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def get_uniform_delta(shape, eps, requires_grad=True):
    """
    Generates a troch uniform random matrix of shape within +-eps.

    Args:
        shape: the tensor shape to create.
        eps: the epsilon bounds 0+-eps for the uniform random tensor.
        requires_grad: whether the tensor requires a gradient.
    """
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta
