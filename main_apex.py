import torch
import torch.distributed as distributed
import model.model_sample as models
import torch.nn as nn
import torch.optim as optim
import argparse
from apex.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import os


def main(args):
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    torch.backends.cudnn.benchmark = True

    # 모델 불러오기
    model = models.ResNet18().cuda()
    ddp_model = DDP(model, delay_allreduce=True)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i in range(2000):
        print(i)
        preds = ddp_model(torch.randn(256, 3, 32, 32).cuda())
        labels = torch.randn(256, 10).cuda()

        loss_function(preds, labels).backward()
        optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
    # automatically by torch.distributed.launch.
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
    # the 'WORLD_SIZE' environment variable will also be set automatically.
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print(args.distributed)

    main(args)