import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group (아무거나 쓰는게 아니다. gloo, mpi, nccl)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method='env://')
    # create local model
    model = nn.Linear(100, 100).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(200, 100).to(rank))
    labels = torch.randn(200, 100).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()


def main():
    world_size = 2
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()