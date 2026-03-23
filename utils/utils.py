import numpy as np
import scipy
import torch
import torchvision
import torch
import torch.nn.functional as F


def get_custom_bright_palette():
    palette = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        # 可以根据您的类别数添加更多颜色
    ], dtype=np.uint8)
    return torch.from_numpy(palette)


def denormalize_for_vis(img_tensor: torch.Tensor):
    device = torch.device("cpu")
    img_tensor_cpu = img_tensor.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return (img_tensor_cpu * std + mean).clamp(0, 1)


def colorize_mask(mask: torch.Tensor, num_classes: int, ignore_index: int = -1):
    # 确保在CPU上执行
    mask_cpu = mask.to(torch.device("cpu"))

    if mask_cpu.dim() == 2:
        # --- 硬标签 (2D) ---
        mask_cpu = mask_cpu.long()
        palette = get_custom_bright_palette()[:num_classes].to(mask_cpu.device)
        colored_mask = torch.zeros(mask_cpu.shape[0], mask_cpu.shape[1], 3, dtype=torch.uint8, device=mask_cpu.device)

        valid_mask = (mask_cpu != ignore_index)
        valid_indices = mask_cpu[valid_mask]

        if valid_indices.numel() > 0:
            valid_indices = valid_indices.clamp(max=num_classes - 1)
            colored_mask[valid_mask] = palette[valid_indices]

        # 将被忽略的像素（值为-1或您定义的其他值）设置为灰色
        colored_mask[~valid_mask] = 128

        return colored_mask.permute(2, 0, 1).float() / 255.0

    elif mask_cpu.dim() == 3 and torch.is_floating_point(mask_cpu):
        # --- 软标签 (3D) ---
        palette = get_custom_bright_palette()[:num_classes].to(mask_cpu.device).float()
        valid_mask = (mask_cpu[0, :, :] != ignore_index)
        valid_probs = mask_cpu * valid_mask.unsqueeze(0).float()
        colored_mask = torch.matmul(valid_probs.permute(1, 2, 0), palette)
        colored_mask[~valid_mask] = 128
        return colored_mask.clamp(0, 255).permute(2, 0, 1).float() / 255.0

    else:
        raise ValueError(f"Unsupported mask format: shape={mask_cpu.shape}, dtype={mask_cpu.dtype}.")


def log_visualizations(writer, global_step, batch_data, num_classes, args):
    """
    将源域和目标域的图像及预测结果记录到TensorBoard。
    此版本与最新的forward函数输出完全对齐。
    """
    # 选择要可视化的样本索引 (通常是batch中的第一个)
    idx_to_show = 0
    ignore_index = getattr(args, 'ignore_index', -1)

    # --- 1. 源域看板 ---
    source_img = batch_data.get('source_img')
    source_lbl = batch_data.get('source_lbl')
    s_ori_logits = batch_data.get('s_logits_ori')
    s_cross_logits = batch_data.get('s_logits_cross')

    # 确保所有必要的源域张量都存在
    if all(tensor is not None for tensor in [source_img, source_lbl, s_ori_logits, s_cross_logits]):
        # 为了安全，确保张量在batch中至少有一个样本
        if source_img.shape[0] > idx_to_show:
            source_img_vis = denormalize_for_vis(source_img[idx_to_show])
            source_lbl_vis = colorize_mask(source_lbl[idx_to_show], num_classes, ignore_index)
            s_ori_pred_vis = colorize_mask(torch.argmax(s_ori_logits[idx_to_show], dim=0), num_classes, ignore_index)
            s_cross_pred_vis = colorize_mask(torch.argmax(s_cross_logits[idx_to_show], dim=0), num_classes,
                                             ignore_index)

            # 您之前的版本有一个 s_lbl_cross_mixed，如果需要，可以按同样方式添加回来
            grid_source = torchvision.utils.make_grid(
                [source_img_vis, source_lbl_vis, s_ori_pred_vis, s_cross_pred_vis],
                nrow=4
            )
            writer.add_image('Vis/1_Source (Img, GT, Pred_Ori, Pred_Cross)', grid_source, global_step)

    # --- 2. 目标域看板 ---
    target_img = batch_data.get('target_img')
    target_logits_student = batch_data.get('target_logits_ori_student')
    target_logits_ema = batch_data.get('target_logits_ema')
    # [关键] 这是由forward函数精心准备的全批次、已处理的硬标签
    pseudo_labels_hard = batch_data.get('pseudo_labels')

    # 确保所有必要的目标域张量都存在
    if all(tensor is not None for tensor in [target_img, target_logits_student, target_logits_ema, pseudo_labels_hard]):
        # 同样，确保张量在batch中至少有一个样本
        if target_img.shape[0] > idx_to_show:
            target_img_vis = denormalize_for_vis(target_img[idx_to_show])

            # 学生模型和教师模型的预测（来自未经筛选的logits）
            t_pred_student_vis = colorize_mask(torch.argmax(target_logits_student[idx_to_show], dim=0), num_classes,
                                               ignore_index)
            t_pred_ema_vis = colorize_mask(torch.argmax(target_logits_ema[idx_to_show], dim=0), num_classes,
                                           ignore_index)

            # 直接使用forward方法生成的最终硬伪标签进行可视化
            # 它已经包含了筛选和类别屏蔽的所有信息，非常直观
            pseudo_labels_vis = colorize_mask(pseudo_labels_hard[idx_to_show], num_classes, ignore_index)

            grid_target = torchvision.utils.make_grid(
                [target_img_vis, t_pred_student_vis, t_pred_ema_vis, pseudo_labels_vis],
                nrow=4
            )
            writer.add_image('Vis/2_Target (Img, Pred_Student, Pred_EMA, Final_PseudoLabel)', grid_target, global_step)

def reverse_mode(mode):
    if mode == "CT":
        return 'MR'
    elif mode == "MR":
        return 'CT'
    elif mode == "ABCT":
        return 'ABMR'
    else:
        return 'ABCT'

