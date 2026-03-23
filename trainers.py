import copy
import gc
import logging
import os
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, cm
from sympy.codegen import Print
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

# from network.PseudoLabelFilter import PseudoLabelProcessor
from utils.utils import log_visualizations
from val import evaluate

logger = logging.getLogger('main_logger')


def to_color(pred):
    """
    将类别标签映射到对应的颜色值。
    Args:
        pred: 预测的标签图，形状为 [B, H, W]，其中每个像素值为 [0.0, 1.0, 2.0, 3.0, 4.0] 之一。
    Returns:
        color_map: 映射后的颜色值图，形状为 [B, H, W]，每个像素值为 [0, 50, 100, 150, 500, 250] 之一。
    """
    # 创建一个映射字典
    intensity_to_color = {
        0.0: 0,
        1.0: 50,
        2.0: 100,
        3.0: 150,
        4.0: 500,
        5.0: 250
    }

    # 将预测标签转换为对应的颜色值
    # 使用torch的tensor操作来映射
    color_map = torch.zeros_like(pred, dtype=torch.float32)

    for intensity, color in intensity_to_color.items():
        # 使用torch的mask来进行替换
        mask = (pred == intensity)
        color_map[mask] = color

    return color_map


def save_label_image(label, output_path, max_labels_per_class=5):
    """
    保存标注图像并在每个类别区域标出像素值。

    Args:
        label (ndarray): 预测结果（形状为 [H, W] 的二维数组）。
        output_path (str): 保存图片的路径。
        max_labels_per_class (int): 每个类别最多标注的像素位置数量。
    """
    plt.figure(figsize=(10, 8))

    # 绘制标注图
    plt.imshow(label, cmap='tab20', interpolation='nearest')
    plt.colorbar()

    # 标注每个类别的像素值
    unique_classes = np.unique(label)
    for cls in unique_classes:
        # 找到每个类别的像素位置
        positions = np.argwhere(label == cls)
        if len(positions) > 0:
            # 随机选择 max_labels_per_class 个位置进行标注
            num_labels = min(len(positions), max_labels_per_class)
            sampled_positions = positions[np.random.choice(len(positions), num_labels, replace=False)]
            for y, x in sampled_positions:
                plt.text(x, y, str(cls), color='white', fontsize=8, ha='center', va='center',
                         bbox=dict(facecolor='black', alpha=0.5))

    # 保存图像
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def sup_train(sup_model, supervised_dataloader, source_val_dataloader, target_val_dataloader,
              sup_opt, seg_loss, num_batch_per_epoch, checkpoint_dir,
              sup_scheduler, args, device):
    global_step = 0
    no_improve_count = 0
    best_val_score = 0
    max_epoch = 0
    sup_model.train()
    checkpoint_name = args.checkpoint_name
    sup_checkpoint_dir = os.path.join(checkpoint_dir, f'{args.stage}', checkpoint_name)

    train_writer = SummaryWriter(
        log_dir=f'../tensorboard/logs/dino/{args.mode}/{args.stage}/{checkpoint_name}')

    for epoch in range(1, args.sup_epochs + 1):
        epoch_loss = 0
        batch_count = 0
        print('checkpoint:', checkpoint_name, args.mode)
        progress_bar = tqdm(total=num_batch_per_epoch, desc=f"Epoch {epoch}/{args.sup_epochs}",
                            unit="batch", ncols=100)
        for supervised_batch in supervised_dataloader:
            torch.cuda.empty_cache()
            s_batch, label = supervised_batch['s'].to(device), supervised_batch['label'].to(device)
            s_pred = sup_model(s_batch)
            s_seg_loss = seg_loss(s_pred, label)
            total_loss = s_seg_loss
            progress_bar.set_postfix(seg=total_loss.item())
            progress_bar.update(1)
            train_writer.add_scalar('SupTrain/total_Loss', total_loss.item(), global_step)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sup_model.parameters(), 1.0)
            sup_opt.step()
            sup_opt.zero_grad()

            global_step += 1
            epoch_loss += total_loss.item()
            batch_count += 1
        sup_scheduler.step()
        current_lr = sup_opt.param_groups[0]['lr']  # 获取当前学习率
        logger.info(f"Epoch [{epoch}], Current Learning Rate: {current_lr}")
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_writer.add_scalar('SupTrain/AvgEpochLoss', avg_epoch_loss, epoch)
        progress_bar.close()

        source_val_score = evaluate(source_val_dataloader, sup_model, num_classes=args.classes)
        target_val_score = evaluate(target_val_dataloader, sup_model, num_classes=args.classes)
        train_writer.add_scalar('Val/source_val_score', source_val_score, epoch)
        train_writer.add_scalar('Val/target_val_score', target_val_score, epoch)

        Path(sup_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        no_improve_count += 1
        val_score = target_val_score
        if val_score > best_val_score:
            best_val_score = val_score
            max_epoch = epoch
            logger.info(f'Max_score is {best_val_score} at epoch{epoch}!')
            save_best_path = os.path.join(sup_checkpoint_dir,
                                          '{}_best_model_{}_{}.pth'.format(args.model_type, epoch,
                                                                           f"{best_val_score:.6f}"))
            best_path = os.path.join(sup_checkpoint_dir, '{}_best_model.pth'.format(args.model_type))
            torch.save(sup_model.state_dict(), save_best_path)
            torch.save(sup_model.state_dict(), best_path)
            no_improve_count = 0  # 重置连续没有改进的周期计数器
        logger.info('best_val_score: %f, epoch: %d', best_val_score, max_epoch)
    train_writer.close()


def get_rampup_weight(epoch, warmup_epochs, max_epochs, max_weight, rampup_percentage=0.4):
    """计算当前epoch的渐进权重 (此函数保持不变)。"""
    if epoch < warmup_epochs:
        return 0.0
    rampup_end_epoch = warmup_epochs + (max_epochs - warmup_epochs) * rampup_percentage
    if epoch >= rampup_end_epoch:
        return max_weight
    progress = (epoch - warmup_epochs) / (rampup_end_epoch - warmup_epochs)
    return max_weight * progress


def _compute_and_log_losses_dsta(losses: Dict[str, torch.Tensor], writer: SummaryWriter, global_step: int, args,
                                 current_weights: Dict[str, float]):
    total_loss_value = 0.0

    # --- 1. 源域监督损失 ---
    # 1a. 原始源域监督
    loss_s_ori = losses.get('seg_s_ori')
    if loss_s_ori is not None:
        weighted_loss = args.lambda_seg * loss_s_ori
        writer.add_scalar('Loss/1a_Supervised_Ori (S)', weighted_loss.item(), global_step)
        total_loss_value += weighted_loss.item()

    # 1c. 跨域对齐监督
    loss_s_cross = losses.get('seg_s_cross')
    if loss_s_cross is not None:
        weighted_loss = args.lambda_seg * loss_s_cross
        writer.add_scalar('Loss/1b_Supervised_Cross (S->T)', weighted_loss.item(), global_step)
        total_loss_value += weighted_loss.item()

    loss_s_stytle_to_t = losses.get('seg_s_style_to_t')
    if loss_s_stytle_to_t is not None:
        weighted_loss = args.lambda_seg * loss_s_stytle_to_t
        writer.add_scalar('Loss/1c_Supervised_Style_to_T (S->T)', weighted_loss.item(), global_step)
        total_loss_value += weighted_loss.item()

    # --- 2. 目标域损失 ---
    # 2b. 伪标签损失
    # [核心修改] 更新损失名称
    loss_pseudo = losses.get('pseudo_t_ori')
    if loss_pseudo is not None:
        weighted_loss = args.lambda_pseudo * loss_pseudo
        writer.add_scalar('Loss/1d_pseudo_t_ori', weighted_loss.item(), global_step)
        total_loss_value += weighted_loss.item()

    writer.add_scalar('Loss/Total', total_loss_value, global_step)

    topk_ratio = losses.get('pl_topk_ratio')
    if topk_ratio is not None:
        writer.add_scalar('PseudoLabel/1_TopK_Ratio', topk_ratio, global_step)

    selected_samples = losses.get('pl_selected_samples')
    if selected_samples is not None:
        writer.add_scalar('PseudoLabel/2_Selected_Samples_Count', selected_samples, global_step)


def unsup_train(model, mixed_dataloader, supervised_dataloader, unsupervised_dataloader, source_val_dataloader,
                target_val_dataloader, opt,
                checkpoint_dir, scheduler, args, device):

    # --- 1. 初始化 ---
    checkpoint_name = args.checkpoint_name
    train_writer = SummaryWriter(
        log_dir=f'../tensorboard/logs/dino/{args.mode}/{args.stage}/{checkpoint_name}')
    unsup_checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_name)
    # 定义预热阶段最佳解码器权重的保存路径
    warmup_weights_path = os.path.join(checkpoint_dir, f"{args.model_type}_best_warmup")
    os.makedirs(unsup_checkpoint_dir, exist_ok=True)
    os.makedirs(warmup_weights_path, exist_ok=True)

    warmup_ckpt_path = os.path.join(warmup_weights_path, f"best_warmup.pth")
    if args.is_sup:
        best_sup_source_val_score = 0.0
        for epoch in range(1, args.epochs + 1):
            gc.collect()
            torch.cuda.empty_cache()
            model.train()
            progress_bar = tqdm(total=len(mixed_dataloader), desc=f"[sup] Epoch {epoch}/{args.epochs}")
            # 预热阶段也使用 mixed_dataloader 以便进行跨域特征混合
            for i_batch, (supervised_batch, unsupervised_batch) in enumerate(mixed_dataloader):
                source_batch = {k: v.to(device) for k, v in supervised_batch.items()}
                opt.zero_grad()
                model_outputs, losses = model(
                    source_batch,
                    None,
                    is_warmup=True,
                    current_iter=0,
                    is_sup = True
                )

                supervised_losses = []
                loss_s_ori = losses.get('seg_s_ori')
                if loss_s_ori is not None:
                    supervised_losses.append(loss_s_ori)
                if not supervised_losses:
                    progress_bar.update(1)
                    continue
                total_loss = args.lambda_seg * torch.stack(supervised_losses).mean()

                if torch.isnan(total_loss) or total_loss.item() == 0:
                    progress_bar.update(1)
                    continue

                total_loss.backward()
                opt.step()

                progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
                progress_bar.update(1)

            progress_bar.close()
            scheduler.step()

            model.eval()
            source_val_score = evaluate(source_val_dataloader, model, num_classes=args.classes, domain_type='target')
            train_writer.add_scalar('Val/Warmup_val_score', source_val_score, epoch)
            print(f"Warm-up Epoch {epoch} | Source Val Score: {source_val_score:.4f}")

            if source_val_score > best_sup_source_val_score:
                best_sup_source_val_score = source_val_score
                warmup_weights_name = os.path.join(warmup_weights_path,
                                                   f"best_sup_{best_sup_source_val_score:.4f}.pth")

                print(
                    f"New best warm-up model found! Score: {best_sup_source_val_score:.4f}. Saving decoder weights...")
                torch.save(model.decoder.state_dict(), warmup_weights_name)

        print(f"\nWarm-up phase finished. Best weights saved to '{warmup_weights_path}'.")
    if not os.path.exists(warmup_ckpt_path) or args.overwrite_warmup_weights:
        print("==========================================================")
        print(f"Warm-up weights not found at '{warmup_weights_path}'.")
        print("Starting WARM-UP phase on source domain.")
        print("==========================================================")

        best_warmup_source_val_score = 0.0

        for epoch in range(1, args.warmup_epochs + 1):
            gc.collect()
            torch.cuda.empty_cache()
            model.train()

            progress_bar = tqdm(total=len(mixed_dataloader), desc=f"[Warm-up] Epoch {epoch}/{args.warmup_epochs}")

            for i_batch, (supervised_batch, unsupervised_batch) in enumerate(mixed_dataloader):
                source_batch = {k: v.to(device) for k, v in supervised_batch.items()}

                target_batch = {k: v.to(device) for k, v in unsupervised_batch.items()}

                opt.zero_grad()
                model_outputs, losses = model(
                    source_batch,
                    target_batch,
                    is_warmup=True,
                    current_iter=0
                )

                supervised_losses = []
                loss_s_ori = losses.get('seg_s_ori')
                if loss_s_ori is not None:
                    supervised_losses.append(loss_s_ori)

                loss_s_cross = losses.get('seg_s_cross')
                if loss_s_cross is not None:
                    supervised_losses.append(loss_s_cross)

                loss_s_style_to_t = losses.get('seg_s_style_to_t')
                if loss_s_style_to_t is not None:
                    supervised_losses.append(loss_s_style_to_t)

                if not supervised_losses:
                    progress_bar.update(1)
                    continue

                total_loss = args.lambda_seg * torch.stack(supervised_losses).mean()

                if torch.isnan(total_loss) or total_loss.item() == 0:
                    progress_bar.update(1)
                    continue

                total_loss.backward()
                opt.step()

                progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
                progress_bar.update(1)

            progress_bar.close()
            scheduler.step()

            model.eval()
            target_val_score = evaluate(target_val_dataloader, model, num_classes=args.classes, domain_type='target')
            train_writer.add_scalar('Val/Warmup_val_score', target_val_score, epoch)
            print(f"Warm-up Epoch {epoch} | Source Val Score: {target_val_score:.4f}")

            if target_val_score > best_warmup_source_val_score:
                best_warmup_source_val_score = target_val_score
                warmup_weights_name = os.path.join(warmup_weights_path,
                                                   f"best_warmup_{best_warmup_source_val_score:.4f}.pth")
                print(
                    f"New best warm-up model found! Score: {best_warmup_source_val_score:.4f}. Saving decoder weights...")
                torch.save(model.decoder.state_dict(), warmup_weights_name)

        print(f"\nWarm-up phase finished. Best weights saved to '{warmup_weights_path}'.")

    print("\n==========================================================")
    print("Proceeding to UDA (Unsupervised Domain Adaptation) phase.")
    print("==========================================================")

    model.load_warmup_weights(warmup_ckpt_path)

    global_step = 0
    best_val_score = 0
    no_improve_count = 0
    max_epoch_saved = 0
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 0.5,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    start_uda_epoch = args.warmup_epochs + 1

    for epoch in range(start_uda_epoch, args.epochs + 1):
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        desc = f"[UDA] Epoch {epoch}/{args.epochs}"
        progress_bar = tqdm(total=len(mixed_dataloader), desc=desc)

        for i_batch, (supervised_batch, unsupervised_batch) in enumerate(mixed_dataloader):
            source_batch = {k: v.to(device) for k, v in supervised_batch.items()}
            target_batch = {k: v.to(device) for k, v in unsupervised_batch.items()}

            opt.zero_grad()

            model_outputs, losses = model(
                source_batch,
                target_batch,
                is_warmup=False,
                current_iter=global_step
            )

            supervised_losses = []
            unsupervised_losses = []

            loss_s_ori = losses.get('seg_s_ori')
            if loss_s_ori is not None: supervised_losses.append(loss_s_ori)
            train_writer.add_scalar('Loss/1a_Supervised_Ori (S)', loss_s_ori.item(), global_step)

            loss_s_cross = losses.get('seg_s_cross')
            if loss_s_cross is not None: supervised_losses.append(loss_s_cross)
            train_writer.add_scalar('Loss/1b_Supervised_Cross (S->T)', loss_s_cross.item(), global_step)

            loss_s_style_to_t = losses.get('seg_s_style_to_t')
            if loss_s_style_to_t is not None: supervised_losses.append(loss_s_style_to_t)
            train_writer.add_scalar('Loss/1c_Supervised_Style_to_T (S->T)', loss_s_style_to_t.item(), global_step)

            loss_pseudo_ori = losses.get('pseudo_t_ori')
            if loss_pseudo_ori is not None: unsupervised_losses.append(loss_pseudo_ori)
            train_writer.add_scalar('Loss/1d_pseudo_t_ori', args.lambda_pseudo * loss_pseudo_ori.item(), global_step)
            total_loss = torch.tensor(0.0, device=device)
            if supervised_losses:
                total_loss += args.lambda_seg * torch.stack(supervised_losses).mean()
            if unsupervised_losses:
                total_loss += args.lambda_pseudo * torch.stack(unsupervised_losses).mean()

            train_writer.add_scalar('Loss/Total', total_loss, global_step)

            topk_ratio = losses.get('pl_topk_ratio')
            if topk_ratio is not None:
                train_writer.add_scalar('PseudoLabel/1_TopK_Ratio', topk_ratio, global_step)

            selected_samples = losses.get('pl_selected_samples')
            if selected_samples is not None:
                train_writer.add_scalar('PseudoLabel/2_Selected_Samples_Count', selected_samples, global_step)
            if torch.isnan(total_loss) or total_loss.item() == 0:
                progress_bar.update(1)
                continue

            total_loss.backward()
            opt.step()

            global_step += 1
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            progress_bar.update(1)

            if global_step % 100 == 0:
                batch_data_for_vis = {
                    'source_img': source_batch['s'],
                    'source_lbl': source_batch['label'],
                    'target_img': target_batch['s'],
                    **model_outputs
                }
                log_visualizations(train_writer, global_step, batch_data_for_vis, model.nclass, args)

        progress_bar.close()
        scheduler.step()

        model.eval()
        source_val_score = evaluate(source_val_dataloader, model, num_classes=args.classes, domain_type='source')
        target_val_score = evaluate(target_val_dataloader, model, num_classes=args.classes, domain_type='target')
        train_writer.add_scalar('Val/source_val_score', source_val_score, epoch)
        train_writer.add_scalar('Val/target_val_score', target_val_score, epoch)
        print(f"Epoch {epoch} | Source Val Score: {source_val_score:.4f} | Target Val Score: {target_val_score:.4f}")

        if target_val_score > best_val_score:
            best_val_score = target_val_score
            max_epoch_saved = epoch
            no_improve_count = 0
            print(f"New best score: {target_val_score:.4f} at epoch {epoch}! Saving model...")
            best_save_path = os.path.join(unsup_checkpoint_dir, f"{args.model_type}_best_model_{best_val_score}.pth")
            torch.save(model.decoder.state_dict(), best_save_path)
            if isinstance(model, torch.nn.DataParallel):
                model.module.update_best_student_weights()
            else:
                model.update_best_student_weights()
        else:
            no_improve_count += 1
            print(
                f"No improvement for {no_improve_count} epochs. Best score: {best_val_score:.4f} at epoch {max_epoch_saved}.")

        if no_improve_count >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if no_improve_count >= 10:
            if isinstance(model, torch.nn.DataParallel):
                model.module._update_ema_epoch_best_weights()
            else:
                model._update_ema_epoch_best_weights()
        if isinstance(model, torch.nn.DataParallel):
            model.module._update_ema_epoch()
        else:
            model._update_ema_epoch()

    train_writer.close()
    print("Training finished.")
    print(f"Best validation score: {best_val_score:.4f} achieved at epoch {max_epoch_saved}.")
