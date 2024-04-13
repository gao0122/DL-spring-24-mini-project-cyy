import argparse


def parse_args_base(parser):
    parser.add_argument('--n', help='network depth', type=int, default=18)
    parser.add_argument('--batch', help='batch size', type=int, default=128)
    parser.add_argument('--dataset_dir', default='./dataset')
    return parser


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser = parse_args_base(parser)
    parser.add_argument('--checkpoint_dir', default='./checkpoint')
    parser.add_argument('--print_freq', help='print loss freq', type=int, default=10)
    parser.add_argument('--save_params_freq', help='parameters saving freq', type=int, default=1000)
    parser.add_argument('--lr', help='initial learning rate', type=float, default=0.1)
    parser.add_argument('--momentum', help='optimizer momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', help='optimizer weight decay (L2 reg.)', type=float, default=0.0005)
    parser.add_argument('--decay_lr_1', help='iteration at which lr decays 1st', type=int, default=24000)
    parser.add_argument('--decay_lr_2', help='iteration at which lr decays 2nd', type=int, default=42000)
    parser.add_argument('--lr_decay_rate', help='lr *= lr_decay_rate at decay_lr_i-th iteration', type=float, default=0.1)
    parser.add_argument('--n_iter', help='learning iterations', type=int, default=30)
    parser.add_argument('--params_path', help='path to saved model weights', default='ckpt/model_final.pth')
    args = parser.parse_args()

    return args
