# Deep Learning Spring 2024

## Mini Project Implementation of ResNet for CIFAR-10 classification

### Requirements

- Python 3.9+
- PyTorch 2.2.0+

### Prerequisites

The script requires several third-party libraries. Below is the list of required packages:

- numpy
- pandas
- matplotlib
- Pillow
- torch
- torchvision

You can install all these packages using pip with the following command:

```bash
pip install numpy pandas matplotlib Pillow torch torchvision
```

### Usage

1. Train and test

```bash
mkdir ckpt
python main.py --n 3 --checkpoint_dir ckpt
```

`n` means the network depth, you can choose from {3, 5, 7, 9}, which means ResNet-{20, 32, 44, 56}.
For other options, please refer helps: `python main.py -h`.
When you run the code for the first time, the dataset will be downloaded automatically.

2. Model parameter file

When your training is done, the model parameter file `ckpt/model_final.pth` will be generated. 

The accuracy test will start right after the training, which will output the test accuracy for reference. 

When the accuracy test is done, the output csv file for the kaggle competition will be generated. 

### Note

If running at HPC, the sbatch could start with for example:

```#!/bin/bash
#SBATCH --job-name=pythonMiniProject
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx8000:1
```

### References

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep residual learning for image recognition," In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.
- Codebase from https://github.com/drgripa1/resnet-cifar10/tree/master