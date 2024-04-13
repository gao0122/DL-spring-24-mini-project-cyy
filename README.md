# Implementation of ResNet for CIFAR-10 classification

## Requirements

- Python 3.9+
- PyTorch 1.6.0+

## Usage

1. Train and test

```
mkdir ckpt
python main.py --n 3 --checkpoint_dir ckpt
```

`n` means the network depth, you can choose from {3, 5, 7, 9}, which means ResNet-{20, 32, 44, 56}.
For other options, please refer helps: `python train.py -h`.
When you run the code for the first time, the dataset will be downloaded automatically.

2. Model parameter file

When your training is done, the model parameter file `path/to/checkpoint_dir/model_final.pth` will be generated.

## Note

If you want to specify GPU to use, you should set environment variable `CUDA_VISIBLE_DEVICES=0`, for example.

## References

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep residual learning for image recognition," In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.