import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import model.model_sample as models
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

## 현재 동작하지 않음 ##


def main(rank, gpu_size):
    distributed.init_process_group(backend='nccl', rank=rank, world_size=gpu_size, init_method='env://')

    model = models.ResNet18()

    ddp_model = DDP(model, device_ids=[rank])

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    preds = ddp_model(torch.randn(1, 3, 32, 32)).to(rank)
    labels = torch.randn(1, 1).to(rank)

    loss_function(preds, labels).backward()
    optimizer.step()


if __name__ == '__main__':
    _gpu_size = torch.cuda.device_count()
    print('gpu_size: ', _gpu_size)
    _rank = 0
    multiprocessing.spawn(main, args=(_rank, ), nprocs=_gpu_size, join=True)
