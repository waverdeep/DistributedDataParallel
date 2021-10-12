import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import model.model_sample as models
import torch.nn as nn
import torch.optim as optim
import argparse
# from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as DDP
import os


def main(args):
    if args.distributed:
        args.gpu = args.local_rank
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.gpu)
        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        distributed.init_process_group(backend='nccl', init_method='env://')

    torch.backends.cudnn.benchmark = True

    model = models.ResNet18().cuda()

    ddp_model = DDP(model, delay_allreduce=True)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    preds = ddp_model(torch.randn(1, 3, 32, 32).cuda())
    labels = torch.randn(1, 1).cuda()

    loss_function(preds, labels).backward()
    optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
    # automatically by torch.distributed.launch.
    parser.add_argument("--local_rank", default=0, type=int)
    _args = parser.parse_args()

    # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
    # the 'WORLD_SIZE' environment variable will also be set automatically.
    _args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        _args.distributed = int(os.environ['WORLD_SIZE']) > 1
    main(_args)