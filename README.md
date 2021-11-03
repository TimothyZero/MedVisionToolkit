## Dependencies
#### Must
- torch >= 1.6.0 (need autocast)

#### Tested 
- g++ 5.4, 7.5
- cuda 9.0, 10.1, 10.2

## Installation

```bash
cd MedToolkit
pip install -r requirement.txt
pip install -e .
```

## Usage

**Train**

```shell
# train with dp on 1 GPU
python tools/train_v2.py --config projects/aneurysm/configs/cfg_baseline_v4_one_v1_large.py
# or on GPUs
python tools/train_v2.py --config projects/aneurysm/configs/cfg_baseline_v4_one_v1_large.py --gpus 2

# train with ddp
python tools/train_v2.py --config projects/aneurysm/configs/cfg_baseline_v4_one_v1_large.py --gpus 2  --dist

```

**Eval**

```shell
python tools/eval.py --config projects/aneurysm/configs/cfg_baseline_v4_one_v3.py --epoch 200 --dataset valid_infer
# then
python projects/aneurysm/evaluation/froc.py --config projects/aneurysm/configs/cfg_baseline_v4_one_v3.py --epoch 200 --dataset valid_infer
# further 
python projects/aneurysm/evaluation/analysis_error.py --config projects/aneurysm/configs/cfg_baseline_v4_one_v3.py --epoch 200 --dataset valid_infer --min 0 --max 10
```

## Features

#### Pipelines

- Support 2d and 3d image

#### CudaDataloader

> dataloader的随机性问题: 虽然index queue具有固定顺序，但是不同worker一个batch完成时间具有随机性，导致data queue输出序列`_try_get_data` and `_get_data`具有随机性，但是`_next_data`确保了顺序，因此在`_process_data`内进行cuda data transform。

- Add `_epoch` in dataloader
- Add Cuda transforms
- Add seed init (corresponds with epoch and index, not the order!) before each batch transforms to make sure **reproducibility**.

#### AnchorGenerator

- Combine anchor and point
- Add patch valid margin
- Add valid index and anchor id

#### AnchorAssiner

- support IoU/MaxIoU/Dist/Uniform

#### DenseHead

- Infer with anchor id to eval in detail
- combine anchor-based head and point-based head together 

#### RoIHead

- support  

#### Runner

- fast switch between DDP and DP 
- easy resume and load
- More human-friendly config  setting
- easy projects creating
- easy inference extraction

#### Reproducibility / Random

- init => model initializationrandom
- sampler/shuffle = > data order random
- worker init => data transform random and cuda transform random
- training assigner and sampler

## TODO

- [x] more powerful and efficient dataloader with:

    - cpu: data loader workers only load data from files \[save GPU memory\]
      
    - gpu: data augmenter do transforms on batch data \[speed up data aug\]
- [ ] replace list index with getattr(), it cannot duplicate in slurm
- [x] speed up inference
  - infer patch sizes
  - float16
  - only backward target prediction
- [x] remove some useless files
- [x] ~~save training images async~~
- [ ] add metric and best saver
- [x] ~~module parent~~
- [x] remove functions in BlockModule
- [x] add monitor in all ComponentModule
- [ ] add new pipeline of training, such as fold, dataset files,
- [ ] compare with mmdet
- [x] ~~dataloader with auto restart~~
- [ ] while no det support 
- [ ] cuda seed


## Attention

- :star: **The model is initialized in the config file, so init the seed before that!** 
- `gc.collect()` cost a lot of time, remove it!
- Using zero padding in convolution layer may cause different behaviour during patch-based inference. Using padding_mode='reflect' instead.
- different run, different epoch, different batch
```python
# 1. runs
# fix all seeds for all runs
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model.init_weights()

# 2.epochs
# set seed in each epoch
# make sure multiprocessing.set_start_method('fork') 
for epoch in epochs:
    np.random.seed(epoch + np.random.get_state()[1][0])

# 3. batches
# set worker_init_fn in dataloader
# workers will be initialized again after loader is over, 
# if persistent_workers is False
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

DataLoader(
    ...,
    worker_init_fn=worker_init_fn
)
```
- For 3d resample, ndi.zoom and skimage.transform.resize will cause different behaviour and output.
- Small structure will lose detail while re-sampling. If the original spacing is (0.7, 0.7, 2.5), reample to (0.7, 0.7, 0.7) maybe better than (1.0, 1.0, 1.0) 


## Common Issues
1. QObject::moveToThread: Current thread is not the object's thread

Solution: reinstall opencv
```shell
pip install opencv-python==4.1.2.30
```


## References

#### framework

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

[MMdetection](https://github.com/open-mmlab/mmdetection)


#### Model

[UNet](https://github.com/milesial/Pytorch-UNet)

[UNet 3D](https://github.com/wolny/pytorch-3dunet)

