# utility functions
from data.facescape_loader import FaceScape
import argparse
import torch
import torchvision
from data.modelnet_loader_torch import ModelNetCls
from src.pctransforms import (
    OnUnitCube,
    PointcloudToTensor,
    PointcloudRotate,
    PointcloudScale,
    PointcloudTranslate,
    PointcloudJitter,
)

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import numpy as np


# fmt: off
def options(argv=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)')  # the result: ${BASENAME}_model_best.pth
    parser.add_argument('--dataset', default="facescape", type=str, help='dataset type: modelnet | facescape')
    parser.add_argument('--annot-file', required=True, type=str, help='path to annotation file relative to base name: facescape/all_annotations1024.npy')
    parser.add_argument('--base-name', required=True, type=str, help='path to parent folder where data is dumped')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing routine. Otherwise, the script will train.')
    parser.add_argument('--sampler', required=True, choices=['fps', 'samplenet', 'random', 'none'], type=str,
                        help='Sampling method.')
    parser.add_argument('--adv-weight', default=50., type=float,
                        help='Weight given to attacker loss')
    parser.add_argument('--scale', default=1., type=float,
                        help='scale for attacker loss')
    parser.add_argument('--transfer-from-pointnet', type=str,
                        metavar='PATH', help='path to trained pointnet 1024')
    parser.add_argument('--transfer-from-dgcnn', type=str,
                        metavar='PATH', help='path to trained dgcnn 1024')
    parser.add_argument('--task', type=str, default="vanilla",
                        help='private (priv) | vanilla')
    parser.add_argument('--base-task', type=str, default="cls_exp", help='cls_exp | cls_age | cls_gender | cls_identity')
    parser.add_argument('--attacker-task', type=str, default='cls_age', help='cls_identity | cls_gender | recon')
    parser.add_argument('--attacker', type=str, metavar='PATH', default='', help='path to pretrained attacker (age)') 
    parser.add_argument('--sampler-model', type=str, metavar='PATH', default='', help='path to pretrained sampler') # use during testing only
    parser.add_argument('--train-pointnet', action='store_true',
                        help='Allow PointNet training.')
    parser.add_argument('--line-cloud', action='store_true',
                        help='Convert to line cloud')
    parser.add_argument('--train-dgcnn', action='store_true',
                        help='Allow DGCNN training.')
    parser.add_argument('--test-dgcnn', action='store_true',
                        help='Allow DGCNN training.')
    parser.add_argument('--train-attacker', action='store_true',
                        help='Allow Attacker training.')
    parser.add_argument('--train-samplenet', action='store_true',
                        help='Allow SampleNet training.')
    parser.add_argument('--use-STN', action='store_true',
                        help='Use spatial transformer to make sampler robust to attack.')
    parser.add_argument('--use-enc-stn', action='store_false', help='Use stn at encoder')
    parser.add_argument('--max-entropy', action='store_true', help='Use max entropy baseline')
    parser.add_argument('--plot-saliency', action='store_true',
                        help='Do eval and plot saliency map in log directiory.')
    parser.add_argument('--contrastive-feat', action='store_true', help='Use contrastive loss for privacy on embeddings of sensitive samples')
    parser.add_argument('--cont-feat-extractor', type=str, default="", help='Cont SN Feature Extractor')
    parser.add_argument('--adv-contrastive', action='store_true', help='Use contrastive loss + Adversarial Training')
    parser.add_argument('--no-adv', action='store_true', help='No Adversarial training with FPS')
    parser.add_argument('--attacker-discrim', action='store_true', help='Do adversarial training with Attacker as the discriminator')
    parser.add_argument('--gaussian', action='store_true', help='Apply gaussian noise')
    parser.add_argument('--resample', action='store_true', help='Learnt noise with resampled sigma')
    parser.add_argument('--learn-noise', action='store_true', help='Learnt noise with FPS')
    parser.add_argument('--cont-scale', type=float, default=1.0, help='Contrastive Loss scale')
    parser.add_argument('--std-noise', type=float, default=1.0, help='Gaussian Noise Sigma')
    parser.add_argument('--reg-weight', type=float, default=5000, help='Regularization Weight')
    parser.add_argument('--visualize', default=0, type=int,
                        help='Number of samples to visualize during test-time')
    parser.add_argument('--finetune', action='store_true', help='Finetune attacker')
    parser.add_argument('--pointwise-dist', action='store_true', help='Learn Noise distribution for each point')

    # settings for on training
    parser.add_argument('--workers', default=16, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=250, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSProp', 'NoamOpt'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained task model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    # Samplenet
    # handled by sputils

    args = parser.parse_args(argv)
    return args

def save_checkpoint(state, filename, suffix):
    torch.save(state, "{}_{}.pth".format(filename, suffix))

def get_datasets(args):
    n = args.num_in_points
    use_contrastive = args.adv_contrastive or args.contrastive_feat
    
    if args.dataset == "facescape":
        transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])
    
        annotation_file = args.annot_file

        if not args.test:
            trainset = FaceScape(
                args.num_in_points,
                transforms=transforms,
                train=True,
                annotations=annotation_file,
                base_directory=args.base_name,
                contrastive=use_contrastive
            )
            testset = FaceScape(
                args.num_in_points,
                transforms=transforms,
                train=False,
                annotations=annotation_file,
                base_directory=args.base_name,
                contrastive=use_contrastive
            )

        else:
            testset = FaceScape(
                args.num_in_points,
                transforms=transforms,
                train=False,
                annotations=annotation_file,
                base_directory=args.base_name,
                contrastive=use_contrastive
            )
            trainset = None
    
    elif args.dataset == "modelnet":
        transforms = torchvision.transforms.Compose(
            [
                PointcloudToTensor(),
                OnUnitCube(),
                PointcloudRotate(axis=np.array([1, 0, 0])),
                PointcloudScale(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ]
        )
        
        test_transforms =  torchvision.transforms.Compose(
            [
                PointcloudToTensor(),
                OnUnitCube()
            ]
        )
        
        if not args.test:            
            trainset = ModelNetCls(
                args.num_in_points,
                transforms,
                train=True,
                contrastive=use_contrastive,
                base_directory=args.base_name
            )
            
            testset = ModelNetCls(
                args.num_in_points,
                test_transforms,
                train=False,
                contrastive=use_contrastive,
                base_directory=args.base_name
            )
            
        else:
            testset = ModelNetCls(
                args.num_in_points,
                test_transforms,
                train=False,
                contrastive=use_contrastive,
                base_directory=args.base_name
            )
            
            trainset=None
        
        
    return trainset, testset
