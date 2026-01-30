import os
import time
import random
import argparse
import configparser
import logging
from datetime import timedelta
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import Getfile
from network.SHAPE_net import SHAPE
from val import evaluate


def get_args():
    parser = argparse.ArgumentParser(description="Testing Script for EVUNet")
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
    parser.add_argument('--model_type', type=str, default='unet2D', choices=['unet2D'],
                        help='Type of the model to use')
    parser.add_argument('--pth_path', type=str, help='mode to run')
    parser.add_argument('--mode', type=str, default="MR", help='mode to run')
    parser.add_argument('--gpu', type=int, default=0, help='gpu to run')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay_rate')
    parser.add_argument('--remind', type=str, default='pre', help='remind me')

    parser.add_argument("--repo_dir", type=str, default="dinov3")
    parser.add_argument("--dino_ckpt", type=str, default="dinov3_checkpoint",
                        help="Path to the pretrained DINO checkpoint (.pth). ")
    parser.add_argument("--dino_size", type=str, default="s", choices=["s+", "s"])

    return parser.parse_args()


def infer_modes_from_path(source_modality):
    if source_modality == 'CT':
        target_modality = 'MR'
    elif source_modality == 'MR':
        target_modality = 'CT'
    elif source_modality == 'ABCT':
        target_modality = 'ABMR'
    elif source_modality == 'ABMR':
        target_modality = 'ABCT'
    else:
        raise ValueError(f"Unknown source modality '{source_modality}' found.")

    return source_modality, target_modality


def load_config(mode, config_path='config_cyc.ini'):
    print(f"Loading config from {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    val_dirs = config.get(mode, 'test_dir')
    val_dir = val_dirs.split(", ")
    label_intensities_str = config.get(mode, 'label_intensities')
    label_intensities = tuple(map(float, label_intensities_str.split(', ')))
    return val_dir, label_intensities


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('main_logger')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    try:
        source_mode, target_mode = infer_modes_from_path(args.mode)
        logger.info(f"Inferred from path: Source Modality='{source_mode}', Target Modality='{target_mode}'")
        logger.info(f"Using config section: '[{source_mode}]'")
    except ValueError as e:
        logger.error(e)
        return

    try:
        val_dir, label_intensities = load_config(source_mode)
        target_val_dir = val_dir[1]
        logger.info(f"Target validation directory for testing: {target_val_dir}")

    except Exception as e:
        logger.error(f"Error loading config or test directories for mode '{source_mode}': {e}")
        return
    logger.info("Loading official DINOv3 backbone weights...")
    if args.dino_size == "s+":
        dino_ckpt = os.path.join(args.dino_ckpt, 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
        backbone = torch.hub.load(repo_or_dir=args.repo_dir, model='dinov3_vits16plus', source='local',
                                  weights=dino_ckpt, pretrained=True)
    elif args.dino_size == "s":
        dino_ckpt = os.path.join(args.dino_ckpt, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        backbone = torch.hub.load(repo_or_dir=args.repo_dir, model='dinov3_vits16', source='local',
                                  weights=dino_ckpt, pretrained=True)
    print(f"Loaded DINOv3 ViT-{args.dino_size.upper()}/16 backbone.")

    logger.info("Initializing EVUNet model structure...")
    model = SHAPE(
        backbone=backbone,
        nclass=args.classes,
        args=args,
    ).to(device)

    checkpoint_full_path = os.path.join(args.checkpoint_path, args.pth_path, 'unet2D_best_model.pth')
    logger.info(f"Loading custom decoder weights from: {checkpoint_full_path}")
    if not os.path.exists(checkpoint_full_path):
        logger.error(f"Checkpoint file not found at {checkpoint_full_path}")
        return

    checkpoint = torch.load(checkpoint_full_path, map_location=device)

    if not isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        else:
            logger.error("Checkpoint file is not a state_dict or a recognizable structure.")
            return

    weights_to_load = {}

    has_decoder_prefix = any(k.startswith('decoder.') for k in checkpoint.keys())
    has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())

    if has_decoder_prefix or has_module_prefix:
        logger.info("Checkpoint format detected: Full model state_dict (possibly with module prefix).")
        for k, v in checkpoint.items():
            clean_key = None
            if '.decoder.' in k:
                clean_key = k.split('.decoder.')[1]
            elif k.startswith('decoder.'):
                clean_key = k.replace('decoder.', '', 1)

            if clean_key is not None:
                weights_to_load[clean_key] = v
    else:
        logger.info("Checkpoint format detected: Decoder-only state_dict.")
        weights_to_load = checkpoint

    if not weights_to_load:
        logger.error("Failed to extract any valid decoder weights from the checkpoint.")
        return

    try:
        missing_keys, unexpected_keys = model.decoder.load_state_dict(weights_to_load, strict=False)
        if unexpected_keys:

            logger.warning(
                f"Unexpected keys in checkpoint that were ignored (not in model's decoder): {unexpected_keys}")

        if missing_keys:
            logger.warning(f"Missing keys in model's decoder not found in checkpoint: {missing_keys}")

        loaded_count = len(model.decoder.state_dict().keys()) - len(missing_keys)
        if loaded_count == 0:
            logger.error("Load failed: No matching keys were found between the checkpoint and the model's decoder.")
            return

        logger.info(f"SUCCESS: Successfully loaded {loaded_count} tensors into the decoder.")

    except Exception as e:
        logger.error(f"An unexpected error occurred while loading weights into the decoder: {e}")
        return

    model.eval()

    logger.info("Initializing DataLoader...")
    test_dataset = Getfile(base_dir=args.data_path, val_dir=target_val_dir, domain=1, num_classes=args.classes,
                           label_intensities=label_intensities, mode=target_mode, onehot=False, num_data=0, aug=False
                           )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"Data loaded. Number of test samples: {len(test_dataset)}")

    logger.info("Start evaluation...")
    start_time = time.time()
    test_score = evaluate(test_dataloader, model, num_classes=args.classes, domain_type='target')
    end_time = time.time()

    logger.info(f"Evaluation finished in {str(timedelta(seconds=end_time - start_time))}")
    logger.info(f"Final Test Score on '{target_mode}' data using '{source_mode}'-trained weights: {test_score:.4f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
