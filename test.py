'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from torch.utils.tensorboard import SummaryWriter

import net
import loss
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='PyTorch Training')


parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')

parser.add_argument('--backbone', default='resnet50', type=str,  # TODO: change this
                    help='bninception, resnet18, resnet34, resnet50, resnet101')
parser.add_argument('--pooling_type', default='GAP', type=str,
                    help='GAP | GMP | GAP,GMP')
parser.add_argument('--input_size', default=224, type=int,
                    help='the size of input batch')

parser.add_argument('--do_nmi', default=True, action='store_true', help='do nmi or not')
parser.add_argument('--freeze_BN', default=True, action='store_true', help='freeze bn')

parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size')
parser.add_argument('--dim', default=512, type=int, help='dimensionality of embeddings')
parser.add_argument('--loss', default='Norm_SoftMax', type=str, help='loss you want')
parser.add_argument('-C', default=1171, type=int, help='C')
parser.add_argument('--data', default='/home/ruofan/PycharmProjects/SoftTriple/datasets/logo2k', help='path to dataset')
parser.add_argument('--data_name', default='logo2k', type=str, help='dataset name')
parser.add_argument('--save_path', default='logs/logo2k_512_NormSoftmax',
                    type=str, help='where your models will be saved')
parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
parser.add_argument('--k_list', default='1,2,4,8', type=str, help='Recall@k list')


def main():
    args = parser.parse_args()
    '''Set number of classes'''
    if args.data_name.lower() in ["car", "cars", "cars196"]:
        args.C = 98
    elif args.data_name.lower() in ["sop", "stanfordonlineproducts"]:
        args.C = 11318
    elif args.data_name.lower() in ["cub", "cub200"]:
        args.C = 100
    elif args.data_name.lower() in ['inshop']:
        args.C = 3997
    else:
        print("Using custom dataset")

    ## create data_loader
    # load data
    testdir = os.path.join(args.data, 'test')

    if 'resnet' in args.backbone:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        scale_value = 1
    else:
        normalize = transforms.Normalize(mean=[104., 117., 128.],
                                         std=[1., 1., 1.])
        scale_value = 255

    test_transforms = transforms.Compose([
        # transforms.Lambda(utils.RGB2BGR),
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(scale_value)),
        normalize, ])
    test_image = datasets.ImageFolder(testdir, test_transforms)

    test_class_dict, max_r = utils.get_class_dict(test_image)
    args.test_class_dict = test_class_dict
    args.max_r = max_r

    test_loader = torch.utils.data.DataLoader(
        test_image,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.data_name.lower() == 'inshop':
        image_info = np.array(test_image.imgs)
        print('\tcheck: gallery == %s, query == %s\n' % (
            image_info[0, 0].split('/')[-3], image_info[-1, 0].split('/')[-3]))
        args.query_labels = np.array(
            [info[0].split('/')[-2] for info in image_info[image_info[:, 1] == '1']])  # 14218 images
        args.gallery_labels = np.array(
            [info[0].split('/')[-2] for info in image_info[image_info[:, 1] == '0']])  # 12612 images
        if len(args.query_labels) != 14218 or len(args.gallery_labels) != 12612:
            print('check you inshop DB')
            exit()

    '''Create model'''
    # define backbone
    if args.backbone == 'bninception':
        model = net.bninception().cuda()
    else:  # resnet family
        model = net.Resnet(resnet_type=args.backbone).cuda()
    # define pooling method
    pooling = net.pooling(pooling_type=args.pooling_type.split(',')).cuda()
    # define embedding method
    embedding = net.embedding(input_dim=model.output_dim, output_dim=args.dim).cuda()

    state_dict = torch.load(os.path.join(args.save_path, 'model_00050.pth'), map_location='cpu')
    model.load_state_dict(state_dict['model_state'])
    embedding.load_state_dict(state_dict['embedding_state'])

    k_list = [int(k) for k in args.k_list.split(',')]  # [1, 2, 4, 8]
    nmi, recall, MAP, features, labels = validate(test_loader, model, pooling, embedding, k_list, args)
    return nmi, recall, MAP

def validate(test_loader, model, pooling, embedding, k_list, args):
    # switch to evaluation mode
    model.eval()
    embedding.eval()

    testdata = torch.Tensor()
    testdata_l2 = torch.Tensor()
    testlabel = torch.LongTensor()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if args.gpu is not None:
                input = input.cuda()
            # compute output
            output = model(input)
            output = pooling(output)
            output = embedding(output)
            output_l2 = F.normalize(output, p=2, dim=1)
            testdata = torch.cat((testdata, output.cpu()), 0)
            testdata_l2 = torch.cat((testdata_l2, output_l2.cpu()), 0)
            testlabel = torch.cat((testlabel, target))

    features = testdata.cpu().numpy().astype('float32')
    features_l2 = testdata_l2.cpu().numpy().astype('float32')
    labels = testlabel.cpu().numpy().astype('float32')
    nmi, recall, MAP = utils.evaluation(features_l2, labels, k_list, args)

    return nmi, recall, MAP, features, labels



if __name__ == '__main__':
    main()
