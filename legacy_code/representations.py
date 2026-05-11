import argparse
import os
import numpy as np
import torch
from models.wrn import build_wideresnet
import torchvision
import pickle

from functools import partial
import pandas as pd
import PIL
import glob

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, io
from torchvision.datasets.utils import verify_str_arg

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', type=str, default='cifar10',
                    help='path to dataset')
parser.add_argument('--lr', type=str, default='1e-4',
                    help='learning rate')
parser.add_argument('--mode', type=str, default='probabilistic-drl',
                    help='Model mode')
parser.add_argument('--aug', type=str, default='none',
                    help='Aug mode')
parser.add_argument('--widen-factor', default=2., type=float, metavar='N',
                    help='widen factor for WReN')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')


channel_stats = {
    'cifar10': dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
    'cifar100': dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
    'mini_imgnet': dict(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                        std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
}

sizes = {
    'cifar10': 32,
    'cifar100': 32,
    'mini_imgnet': 84
}

padding = {
    'cifar10': 4,
    'cifar100': 4,
    'mini_imgnet': 8
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_dataloaders(dataset):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(**channel_stats[dataset])])

    if dataset == 'cifar10':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar10/by-image/train+val',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar10/by-image/test',
            transform = transform
        )
    elif dataset == 'cifar100':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar100/by-image/train+val',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar100/by-image/test',
            transform = transform
        )
    elif dataset == 'mini_imgnet':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/miniimagenet/train',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/miniimagenet/test',
            transform = transform
        )
    else:
        print('Dataset Not Supported')
        exit()

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = 1000,
        shuffle=False,
        num_workers = 1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = 1000,
        shuffle=False,
        num_workers = 1
    )

    return train_loader, test_loader


def main():
    args = parser.parse_args()
    if args.widen_factor.is_integer():
        args.widen_factor = int(args.widen_factor)

    name = f'trained_models/{args.lr}/{args.mode}/{args.data}_{args.widen_factor}_{args.aug}_{args.seed}'

    for random in [False]:
        if random:
            tr_name = 'random_train'
            t_name = 'random_test'
        else:
            tr_name = 'train'
            t_name = 'test'

        train, test = get_dataloaders(args.data)

        if args.data == 'cifar10':
            num_classes = 10
        else:
            num_classes = 100

        if 'vae' in args.mode or 'probabilistic' in args.mode:
            prob = True
        else:
            prob = False

        if args.widen_factor < 1e-8:
            latent_dim = 2
        else:
            latent_dim = int(64 * args.widen_factor)

        model = build_wideresnet(28, args.widen_factor, 0, num_classes, latent_dim, prob)
        print(model)

        if not random:
            state_dict = torch.load(f'{name}/checkpointsenc_/encoder_state_1.pth')
            for k in list(state_dict.keys()):
                if k == 'linear.weight' or k == 'linear.bias':
                    del state_dict[k]
            log = model.load_state_dict(state_dict, strict=False)
            assert log.missing_keys == ['linear.weight', 'linear.bias']
            print('Model State Loaded')

        model = model.cuda()
        model.eval()

        for granularity in [2, 5, 7, 10, 17, 25, 37, 50]:
            print(f'Granularity: {granularity}')
            test_name = f'{t_name}_{granularity}.pkl'
            train_name = f'{tr_name}_{granularity}.pkl'

            if os.path.exists(f'{name}/{test_name}'):
                print("Representation Exists")
                continue

            times = (np.arange(granularity+1) / granularity)

            enc_x, enc_y = get_encoding(model, train, times, prob)
            with open(f'{name}/{train_name}', 'wb') as f:
                pickle.dump({
                    "Representation": enc_x,
                    "Labels": enc_y
                }, f, pickle.HIGHEST_PROTOCOL)

            enc_x, enc_y = get_encoding(model, test, times, prob)
            with open(f'{name}/{test_name}', 'wb') as f:
                pickle.dump({
                    "Representation": enc_x,
                    "Labels": enc_y
                }, f, pickle.HIGHEST_PROTOCOL)


def get_encoding(model, data, times, prob):
    enc_x = dict()
    enc_y = []

    for time in times:
        enc_x[time] = []

    print(len(data))

    for i, (x, y) in enumerate(data):
        print(i)
        enc_y.append(y.detach().cpu().numpy())

        x = x.cuda()
        for time in times:
            tm = torch.tensor(time).view(1).repeat(x.shape[0]).cuda().float()
            with torch.no_grad():
                _, _, enc = model(x, tm)
            if prob:
                enc, _ = torch.split(enc, enc.shape[-1] // 2, dim=-1)
            enc_x[time].append(enc.detach().cpu().numpy())

    enc_y = np.reshape(np.concatenate(enc_y, axis=0), (-1))
    for time in times:
        enc_x[time] = np.concatenate(enc_x[time], axis=0)

    return enc_x, enc_y

if __name__ == "__main__":
    main()
