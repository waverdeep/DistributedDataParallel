# DistributedDataParallel
모델 학습 시간 단축 및 GPU 메모리를 효율적으로 (여러개의 GPU 메모리에 아주 꽉꽉 눌러담아) 사용하기 위해 Baseline code 만들기 
- torch.nn.parallel.DistributedDataParallel
- apex.parallel.DistributedDataParallel

## Getting Started
- torch.nn.parallel.DistributedDataParallel 을 이용한 구현 
```angular2html
python main_DDP.py
```
- apex.parallel.DistributedDataParallel 을 이용한 구현 
```
python -m torch.distributed.launch --nproc_per_node=YOUR_GPU_COUNT main_apex.py --local_rank 2
```

### Prerequisites
- Install Nvidia apex (만약 apex를 사용한 DistributedDataParallel 을 사용하기 위해서라면) [NVIDIA APEX DOWNLOAD](https://github.com/NVIDIA/apex)


### Installing

- Latest Pytorch [INSTALL PYTORCH](https://pytorch.org/get-started/locally/)
- some python modules...

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details
