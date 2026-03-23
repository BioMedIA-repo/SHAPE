import argparse
import itertools
import torch
import os

from network.SHAPE_net import SHAPE

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
parser.add_argument('--data_path', type=str, default='../data2D', help='Name of Experiment')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Name of Experiment')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help="[Stage 1] Number of epochs for Generator's unsupervised pre-training.")

parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-4,
                    help='Main training learning rate', dest='lr')
parser.add_argument('--load', '-f', type=int, default=0, help='Load model from a .pth file')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='save_checkpoint')

parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
parser.add_argument('--seed', type=int, default=8888, help='random seed')
parser.add_argument('--patience', type=int, default=200, help='patience for student model')
parser.add_argument('--model_type', type=str, default='shape',
                    help='Type of the model to use')
parser.add_argument('--mode', type=str, default="CT", help='mode to run')
parser.add_argument('--gpu', type=int, default=1, help='gpu to run')
parser.add_argument('--decay_rate', type=float, default=0.95, help='decay_rate')

parser.add_argument('--stage', type=str, default='train', choices=['sup', 'unsup'], help='stage')
parser.add_argument('--checkpoint_name', type=str, default='checkpoint_name', help='checkpoint_name')
parser.add_argument("--repo_dir", type=str, default="dinov3")
parser.add_argument("--dino_ckpt", type=str, default="dinov3_checkpoint",
                    help="Path to the pretrained DINO checkpoint (.pth). "
                         "Use ViT-B/16 checkpoint for --dino_size b, "
                         "or ViT-S/16 checkpoint for --dino_size s.")
parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"],
                    help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")

parser.add_argument('--overwrite_warmup_weights', action='store_true',
                    help='Enable pseudo-label self-training.')

parser.add_argument('--lambda_seg', type=float, default=1.0,
                    help="Weight for the supervised segmentation loss on the source domain's normal path.")


parser.add_argument('--use_hfm', action='store_true',
                    help='Enable the Hierarchical Feature Modulation (HFM) module. '
                         'The model will use original DINO features directly.')
parser.add_argument('--use_selector', action='store_true',
                    help='Enable the Hypergraph consistency loss on target domain.')
parser.add_argument('--use_refinement', action='store_true',
                    help='Enable pseudo-label refinement.')
parser.add_argument('--use_pseudo_labels', action='store_true',
                    help='Enable pseudo-label self-training.')

parser.add_argument('--hpe_fusion_alpha', type=float, default=0.25,
                    help='[Hyperparam] The fusion weight alpha for shape and layout scores in HPE 超图类内和类间加权比例.')

parser.add_argument('--selector_initial_k', type=float, default=0.1,
                    help='[Hyperparam] The initial top-k percentile for the pseudo-label selector.')

parser.add_argument('--sap_threshold_percentile', type=float, default=50.0,
                    help='[Hyperparam] The q-th percentile to set the anomaly threshold for SAP named theta.')

parser.add_argument('--lambda_pseudo', type=float, default=1,
                    help="Max weight for the pseudo-label self-training loss on the target domain.")

parser.add_argument('--pure_tao', type=float, default=1,
                    help="tao.")
parser.add_argument("--is_sup", type=bool, default=False,)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['OMP_NUM_THREADS'] = str(1)

import ast
import configparser
import copy
import json
import logging
import random
import sys
import time
from ast import literal_eval
from datetime import timedelta

import monai
import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss, CosineSimilarity
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss

from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from utils.utils import reverse_mode
from dataloader import Getfile, MixedDataLoader

from trainers import sup_train, unsup_train


def main(args):
    total_start_time = time.time()
    learning_rate = args.lr
    data_path = args.data_path
    batch_size = args.batch_size
    num_classes = args.classes
    save_checkpoint = args.save_checkpoint
    dir_checkpoint = os.path.join(args.checkpoint_path, args.mode)

    config = configparser.ConfigParser()
    print("Loading config.ini")
    config.read('config.ini')
    train_dirs_str = config.get(args.mode, 'train_dirs')
    train_dirs = ast.literal_eval(train_dirs_str)
    val_dirs = config.get(args.mode, 'test_dir')
    val_dir = val_dirs.split(", ")
    label_intensities_str = config.get(args.mode, 'label_intensities')
    label_intensities = tuple(map(float, label_intensities_str.split(', ')))
    class_to_pixel_str = config.get(args.mode, 'class_to_pixel')
    class_to_pixel = json.loads(class_to_pixel_str)

    try:
        if_aug = True
        if_vision = False
        sup_num_data = 8000
        unsup_num_data = 8000
        if_shuffle = True
        supervised_ratio = 1
        sup_batch_size = batch_size
        unsup_batch_size = batch_size
        if args.stage == 'unsup':
            supervised_ratio = 0.5
            sup_batch_size = int(batch_size * supervised_ratio)
            unsup_batch_size = batch_size - sup_batch_size

        start_time = time.time()
        supervised_data = Getfile(base_dir=data_path, image_dirs=train_dirs, domain=0, num_classes=num_classes,
                                  label_intensities=label_intensities, mode=args.mode, onehot=True,
                                  num_data=sup_num_data,
                                  aug=if_aug, vision=if_vision)
        supervised_dataloader = DataLoader(supervised_data, batch_size=sup_batch_size, shuffle=if_shuffle,
                                           num_workers=12, persistent_workers=True, drop_last=True)

        unsupervised_data = Getfile(base_dir=data_path, image_dirs=train_dirs, domain=1, num_classes=num_classes,
                                    label_intensities=label_intensities, mode=reverse_mode(args.mode), onehot=True,
                                    num_data=unsup_num_data, aug=if_aug, vision=if_vision)
        unsupervised_dataloader = DataLoader(unsupervised_data, batch_size=unsup_batch_size, shuffle=if_shuffle,
                                             num_workers=12, persistent_workers=True, drop_last=True)
        # 创建混合数据加载器
        mixed_dataloader = MixedDataLoader(supervised_dataloader, unsupervised_dataloader,
                                           supervised_ratio=supervised_ratio)

        source_val_dataloader = DataLoader(
            Getfile(base_dir=data_path, val_dir=val_dir[0], domain=1, num_classes=num_classes,
                    label_intensities=label_intensities, mode=args.mode, onehot=False, num_data=0, aug=False),
            batch_size=16, shuffle=False, num_workers=8)
        target_val_dataloader = DataLoader(
            Getfile(base_dir=data_path, val_dir=val_dir[1], domain=1, num_classes=num_classes,
                    label_intensities=label_intensities, mode=args.mode, onehot=False, num_data=0, aug=False),
            batch_size=16, shuffle=False, num_workers=8)
        end_time = time.time()
        print(f"Data loaded in {str(timedelta(seconds=end_time - start_time))}")

        sup_segloss = monai.losses.DiceFocalLoss(softmax=True)
        seg_loss = DiceFocalLoss(softmax=True)
        mse_loss = MSELoss()
        ################################################################################################################
        # sup train and unsup train
        ################################################################################################################
        if args.dino_size == "b":  # 实际是S+
            dino_ckpt = os.path.join(args.dino_ckpt, 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
            backbone = torch.hub.load(repo_or_dir=args.repo_dir, model='dinov3_vits16plus', source='local',
                                      weights=dino_ckpt, pretrained=True)
            encoder_size = 'base'
        else:
            dino_ckpt = os.path.join(args.dino_ckpt, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
            backbone = torch.hub.load(repo_or_dir=args.repo_dir, model='dinov3_vits16', source='local',
                                      weights=dino_ckpt, pretrained=True)
            encoder_size = 'small'
        if hasattr(backbone, 'blocks'):
            num_encoder_layers = len(backbone.blocks)
            print(f"编码器的层数是: {num_encoder_layers}")
        else:
            print("在模型结构中未找到 'blocks' 模块。")
        print(f"Loaded DINOv3 ViT-{args.dino_size.upper()}/16 backbone.")

        model = SHAPE(
            backbone=backbone,
            nclass=args.classes,
            args=args,
        ).to(device)

        if args.stage == 'sup':
            print('start supervised training')
            model.train()

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
            trainable_params = [
                {'params': model.module.geometric_extractor.parameters()},
                {'params': model.module.guidance_generator.parameters()},
                {'params': model.module.detail_refiner.parameters()}
            ]

            sup_opt = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=1e-4
            )
            sup_scheduler = CosineAnnealingLR(sup_opt, T_max=args.sup_epochs, eta_min=1e-6)
            num_batch_per_epoch = len(supervised_dataloader)
            sup_train(model, supervised_dataloader, source_val_dataloader, target_val_dataloader,
                      sup_opt, sup_segloss, num_batch_per_epoch, dir_checkpoint,
                      sup_scheduler, args, device)
        elif args.stage == 'unsup':
            print('start unsupervised training')
            model.train()

            module_grad_status = {}
            for name, param in model.named_parameters():
                module_name = name.split('.')[0]
                current_status = module_grad_status.get(module_name, False)
                module_grad_status[module_name] = current_status or param.requires_grad
            print("\n--- Summary of Module Trainable Status ---")
            max_len = max(len(name) for name in module_grad_status.keys()) if module_grad_status else 0
            for module_name, is_trainable in module_grad_status.items():
                print(f"  - Module: {module_name:<{max_len}}   Trainable: {is_trainable}")
            print("----------------------------------------\n")
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            unsup_train(
                model=model,
                mixed_dataloader=mixed_dataloader,
                supervised_dataloader=supervised_dataloader,
                unsupervised_dataloader=unsupervised_dataloader,
                source_val_dataloader=source_val_dataloader,
                target_val_dataloader=target_val_dataloader,
                opt=optimizer,
                checkpoint_dir=os.path.join(args.checkpoint_path, args.mode),
                scheduler=scheduler,
                args=args,
                device=device
            )

        total_end_time = time.time()
        print(f"Total training time: {str(timedelta(seconds=total_end_time - total_start_time))} seconds")

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    log_file = 'training_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('main_logger')
    logger.addHandler(file_handler)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    device_str = f'cuda:{gpu_device}' if gpu_device is not None else 'cpu'
    logger.info(f'Using device {device_str}')

    # 配置cuDNN
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(args)
