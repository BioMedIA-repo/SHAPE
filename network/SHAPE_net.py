from copy import deepcopy
import math
from typing import List, Tuple, Dict, Callable, Any, Optional, Literal
import types
from monai.losses import DiceFocalLoss
from collections.abc import Callable, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from torch.nn.modules.loss import _Loss
from monai.losses.focal_loss import FocalLoss
from monai.networks import one_hot
from network.PseudoLabelFilter import RefiningSelector

features_storage = {}


def get_capture_attention_fn(original_attention_module: nn.Module) -> Callable:
    scale = original_attention_module.scale

    def new_compute_attention(self, qkv, attn_bias=None, rope=None):
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv_reshaped = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv_reshaped, 2)

        q_for_storage = q.transpose(1, 2).reshape(B, N, C)
        k_for_storage = k.transpose(1, 2).reshape(B, N, C)

        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        attn_map_softmax = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)

        module_id = id(self)
        features_storage[module_id] = {
            'q': q_for_storage.detach(),
            'k': k_for_storage.detach(),
            'attn_map': attn_map_softmax.detach()
        }

        x = (attn_map_softmax @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    return new_compute_attention


class MultiStrategyTokenGenerationHD(nn.Module):
    def __init__(self,
                 num_classes: int,
                 purity_threshold: float = 0.9,
                 confidence_threshold: float = 0.9,
                 intra_domain_replace_ratio: float = 0.3,
                 pseudo_label_weight: float = 1.0,
                 patch_mix_strategy: Literal['swap', 'mixup'] = 'mixup',
                 mixup_alpha_range: tuple = (0.4, 0.6),
                 ignore_index: int = -1,
                 proto_momentum: float = 0.99):  # 新增动量参数
        super().__init__()
        self.num_classes = num_classes
        self.purity_threshold = purity_threshold
        self.confidence_threshold = confidence_threshold
        self.intra_domain_replace_ratio = intra_domain_replace_ratio
        self.pseudo_label_weight = pseudo_label_weight
        self.ignore_index = ignore_index
        assert patch_mix_strategy in ['swap', 'mixup'], \
            f"patch_mix_strategy must be either 'swap' or 'mixup', but got {patch_mix_strategy}"
        self.patch_mix_strategy = patch_mix_strategy
        self.mixup_alpha_range = mixup_alpha_range

        self.register_buffer('s_prototypes_ema', torch.zeros(num_classes, 0))
        self.register_buffer('t_prototypes_ema', torch.zeros(num_classes, 0))
        self.register_buffer('proto_initialized', torch.tensor(False))
        self.proto_momentum = proto_momentum

    @torch.no_grad()
    def forward(self,
                s_feat_map: torch.Tensor,
                t_feat_map: torch.Tensor,
                s_label_pixel: torch.Tensor,
                t_probs_pixel: torch.Tensor
                ) -> Dict[str, torch.Tensor]:

        original_s_shape = s_feat_map.shape
        original_t_shape = t_feat_map.shape
        s_feat_map_fine = F.interpolate(s_feat_map, scale_factor=2, mode='bilinear', align_corners=False)
        t_feat_map_fine = F.interpolate(t_feat_map, scale_factor=2, mode='bilinear', align_corners=False)

        B, C_feat, H_p, W_p = s_feat_map_fine.shape
        N = H_p * W_p

        if not self.proto_initialized:
            self.s_prototypes_ema = torch.zeros(self.num_classes, C_feat, device=s_feat_map.device)
            self.t_prototypes_ema = torch.zeros(self.num_classes, C_feat, device=t_feat_map.device)
            self.proto_initialized.fill_(True)
        elif self.s_prototypes_ema.shape[1] != C_feat:
            self.s_prototypes_ema = torch.zeros(self.num_classes, C_feat, device=s_feat_map.device)
            self.t_prototypes_ema = torch.zeros(self.num_classes, C_feat, device=t_feat_map.device)

        img_h, img_w = s_label_pixel.shape[1], s_label_pixel.shape[2]
        s_feat_token = s_feat_map_fine.flatten(2).transpose(1, 2).contiguous()
        t_feat_token = t_feat_map_fine.flatten(2).transpose(1, 2).contiguous()
        patch_h, patch_w = img_h // H_p, img_w // W_p

        t_pred_conf_pixel, t_pred_label_pixel = torch.max(t_probs_pixel, dim=1)
        t_pred_label_filtered = t_pred_label_pixel.clone()
        t_pred_label_filtered[t_pred_conf_pixel < self.confidence_threshold] = self.ignore_index
        s_label_patches_hard = F.unfold(s_label_pixel.unsqueeze(1).float(), kernel_size=(patch_h, patch_w),
                                        stride=(patch_h, patch_w)).transpose(1, 2).contiguous().long()
        t_label_patches_hard = F.unfold(t_pred_label_filtered.unsqueeze(1).float(), kernel_size=(patch_h, patch_w),
                                        stride=(patch_h, patch_w)).transpose(1, 2).contiguous().long()
        s_probs_pixel = F.one_hot(s_label_pixel, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        s_probs_unfolded = F.unfold(s_probs_pixel, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        t_probs_unfolded = F.unfold(t_probs_pixel, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

        s_feat_cross, t_feat_cross = s_feat_token.clone(), t_feat_token.clone()
        s_label_cross_unfolded_soft = s_probs_unfolded.clone()
        t_label_cross_unfolded_soft = t_probs_unfolded.clone()

        for b in range(B):
            def get_pure_and_impure_tokens(label_patches_b, conf_pixel_b=None):
                patches = label_patches_b.reshape(N, -1)
                mode_labels, _ = torch.mode(patches, dim=-1)
                valid_pixels_mask = (patches != self.ignore_index)
                num_valid_pixels = valid_pixels_mask.sum(dim=-1)
                purity = ((patches == mode_labels.unsqueeze(-1)) & valid_pixels_mask).sum(
                    dim=-1) / num_valid_pixels.clamp(min=1.0)

                pure_mask = (purity >= self.purity_threshold) & (num_valid_pixels > 0)
                if conf_pixel_b is not None:
                    conf_token_b = F.avg_pool2d(conf_pixel_b.unsqueeze(0).unsqueeze(0), kernel_size=(patch_h, patch_w),
                                                stride=(patch_h, patch_w)).reshape(N)
                    pure_mask &= (conf_token_b >= self.confidence_threshold)

                pure_indices = torch.where(pure_mask)[0]
                impure_indices = torch.where(~pure_mask)[0]
                pure_labels = mode_labels[pure_indices]

                return pure_indices, pure_labels, impure_indices

            s_pure_indices, s_pure_labels, s_impure_indices = get_pure_and_impure_tokens(s_label_patches_hard[b])
            t_pure_indices, t_pure_labels, t_impure_indices = get_pure_and_impure_tokens(t_label_patches_hard[b],
                                                                                         t_pred_conf_pixel[b])

            alpha = -1
            if self.patch_mix_strategy == 'mixup':
                alpha_low, alpha_high = self.mixup_alpha_range
                alpha = alpha_low + torch.rand(1).item() * (alpha_high - alpha_low)

            if s_pure_indices.numel() > 0:
                s_pure_feats = s_feat_token[b, s_pure_indices]

                for c in range(self.num_classes):
                    if (s_mask := s_pure_labels == c).any():
                        current_mean = s_pure_feats[s_mask].mean(0)
                        self.s_prototypes_ema[c] = self.proto_momentum * self.s_prototypes_ema[c] + \
                                                   (
                                                               1.0 - self.proto_momentum) * current_mean.detach()  # detach 防止梯度回传到 buffer

                s_pure_distances = torch.norm(s_pure_feats - self.s_prototypes_ema[s_pure_labels], dim=-1)

                if t_pure_indices.numel() > 0:
                    t_pure_feats = t_feat_token[b, t_pure_indices]

                    for c in range(self.num_classes):
                        if (t_mask := t_pure_labels == c).any():
                            current_mean = t_pure_feats[t_mask].mean(0)
                            self.t_prototypes_ema[c] = self.proto_momentum * self.t_prototypes_ema[c] + \
                                                       (1.0 - self.proto_momentum) * current_mean.detach()

                    t_pure_distances = torch.norm(t_pure_feats - self.t_prototypes_ema[t_pure_labels], dim=-1)

                    for c in range(self.num_classes):
                        s_class_indices = torch.where(s_pure_labels == c)[0]
                        t_class_indices = torch.where(t_pure_labels == c)[0]
                        s_num, t_num = len(s_class_indices), len(t_class_indices)

                        if s_num > 0 and t_num > 0:
                            target_idx_s = s_pure_indices[s_class_indices]
                            t_class_dists = t_pure_distances[t_class_indices]
                            source_pool_idx_t = t_pure_indices[t_class_indices[torch.argsort(t_class_dists)]]
                            source_idx_t = source_pool_idx_t.repeat(math.ceil(s_num / t_num))[:s_num]

                            target_idx_t = t_pure_indices[t_class_indices]
                            s_class_dists = s_pure_distances[s_class_indices]
                            source_pool_idx_s = s_pure_indices[s_class_indices[torch.argsort(s_class_dists)]]
                            source_idx_s = source_pool_idx_s.repeat(math.ceil(t_num / s_num))[:t_num]

                            mixed_feats_s = alpha * s_feat_token[b, target_idx_s] + (1.0 - alpha) * \
                                            t_feat_token[b, source_idx_t]
                            mixed_feats_t = alpha * t_feat_token[b, target_idx_t] + (1.0 - alpha) * \
                                            s_feat_token[b, source_idx_s]

                            s_feat_cross[b, target_idx_s] = mixed_feats_s
                            s_label_cross_unfolded_soft[b, :, target_idx_s] = t_probs_unfolded[
                                                                                  b, :, source_idx_t] * self.pseudo_label_weight

                            t_feat_cross[b, target_idx_t] = mixed_feats_t
                            t_label_cross_unfolded_soft[b, :, target_idx_t] = s_probs_unfolded[b, :, source_idx_s]

            if self.patch_mix_strategy == 'mixup' and s_impure_indices.numel() > 0 and t_impure_indices.numel() > 0:
                s_impure_feats = s_feat_token[b, s_impure_indices]
                t_impure_feats = t_feat_token[b, t_impure_indices]
                eps = 1e-5

                if s_impure_feats.shape[0] > 1:
                    s_content_mean, s_content_std = s_impure_feats.mean(dim=0), s_impure_feats.std(dim=0)
                else:
                    s_content_mean, s_content_std = s_impure_feats.mean(dim=0), torch.zeros_like(
                        s_impure_feats.mean(dim=0))

                if t_impure_feats.shape[0] > 1:
                    t_content_mean, t_content_std = t_impure_feats.mean(dim=0), t_impure_feats.std(dim=0)
                else:
                    t_content_mean, t_content_std = t_impure_feats.mean(dim=0), torch.zeros_like(
                        t_impure_feats.mean(dim=0))

                s_style_mean, s_style_std = t_content_mean, t_content_std
                t_style_mean, t_style_std = s_content_mean, s_content_std

                mixed_mean_for_s = alpha * s_content_mean + (1.0 - alpha) * s_style_mean
                mixed_std_for_s = alpha * s_content_std + (1.0 - alpha) * s_style_std
                mixed_mean_for_t = alpha * t_content_mean + (1.0 - alpha) * t_style_mean
                mixed_std_for_t = alpha * t_content_std + (1.0 - alpha) * t_style_std

                if s_impure_feats.shape[0] > 0:
                    s_var = s_impure_feats.var(dim=0, unbiased=False)
                    s_feat_normalized = (s_impure_feats - s_content_mean.unsqueeze(0)) / torch.sqrt(
                        s_var.unsqueeze(0) + eps)
                    s_feat_stylized = s_feat_normalized * mixed_std_for_s.unsqueeze(0) + mixed_mean_for_s.unsqueeze(0)
                    s_feat_cross[b, s_impure_indices] = s_feat_stylized

                if t_impure_feats.shape[0] > 0:
                    t_var = t_impure_feats.var(dim=0, unbiased=False)
                    t_feat_normalized = (t_impure_feats - t_content_mean.unsqueeze(0)) / torch.sqrt(
                        t_var.unsqueeze(0) + eps)
                    t_feat_stylized = t_feat_normalized * mixed_std_for_t.unsqueeze(0) + mixed_mean_for_t.unsqueeze(0)
                    t_feat_cross[b, t_impure_indices] = t_feat_stylized

        s_cross_fine = s_feat_cross.transpose(1, 2).view_as(s_feat_map_fine)
        t_cross_fine = t_feat_cross.transpose(1, 2).view_as(t_feat_map_fine)
        s_cross_restored = F.interpolate(s_cross_fine, size=original_s_shape[-2:], mode='bilinear', align_corners=False)
        t_cross_restored = F.interpolate(t_cross_fine, size=original_t_shape[-2:], mode='bilinear', align_corners=False)

        s_stylized_t_fine = adaptive_instance_normalization(s_feat_map_fine, t_feat_map_fine)
        t_stylized_s_fine = adaptive_instance_normalization(t_feat_map_fine, s_feat_map_fine)
        s_stylized_t_restored = F.interpolate(s_stylized_t_fine, size=original_s_shape[-2:], mode='bilinear',
                                              align_corners=False)
        t_stylized_s_restored = F.interpolate(t_stylized_s_fine, size=original_t_shape[-2:], mode='bilinear',
                                              align_corners=False)

        s_cross_label_pixel_soft = F.fold(s_label_cross_unfolded_soft, (img_h, img_w), (patch_h, patch_w),
                                          stride=(patch_h, patch_w))
        t_cross_label_pixel_soft = F.fold(t_label_cross_unfolded_soft, (img_h, img_w), (patch_h, patch_w),
                                          stride=(patch_h, patch_w))

        s_cross_label_weights, _ = torch.max(s_cross_label_pixel_soft, dim=1)
        not_mixed_mask_unfolded = (s_label_cross_unfolded_soft == s_probs_unfolded).all(dim=1)
        not_mixed_mask_patch_grid = not_mixed_mask_unfolded.view(B, 1, H_p, W_p)
        not_mixed_mask_pixel = F.interpolate(not_mixed_mask_patch_grid.float(), size=(img_h, img_w),
                                             mode='nearest').bool().squeeze(1)
        s_cross_label_weights[not_mixed_mask_pixel] = 1.0
        t_cross_label_weights, _ = torch.max(t_cross_label_pixel_soft, dim=1)

        output = {
            "s_cross": s_cross_restored,
            "t_cross": t_cross_restored,
            "s_cross_label_pixel": s_cross_label_pixel_soft,
            "s_cross_label_weights": s_cross_label_weights,
            "t_cross_label_pixel": t_cross_label_pixel_soft,
            "t_cross_label_weights": t_cross_label_weights,
            "s_stylized_t": s_stylized_t_restored,
            "t_stylized_s": t_stylized_s_restored,
        }
        return output


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):

    B, C, H, W = content_feat.shape

    content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    content_var = content_feat.var(dim=[2, 3], keepdim=True, unbiased=False)
    content_std = torch.sqrt(content_var + eps)

    style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    style_var = style_feat.var(dim=[2, 3], keepdim=True, unbiased=False)
    style_std = torch.sqrt(style_var + eps)

    normalized_feat = (content_feat - content_mean) / content_std
    stylized_feat = normalized_feat * style_std + style_mean

    return stylized_feat


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        num_groups = 32 if out_ch % 32 == 0 else 16 if out_ch % 16 == 0 else 8 if out_ch % 8 == 0 else 1

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, n_class, embed_dim, aux_channels=1):
        super().__init__()
        decoder_channels = [embed_dim // 2, embed_dim // 4, embed_dim // 8, embed_dim // 16]

        bottleneck_in_ch_with_aux = embed_dim + aux_channels
        self.bottleneck_with_aux = ConvBlock(bottleneck_in_ch_with_aux, embed_dim)
        self.bottleneck_no_aux = ConvBlock(embed_dim, embed_dim)

        self.up_conv1 = ConvBlock(embed_dim, decoder_channels[0])
        self.up_conv2 = ConvBlock(decoder_channels[0], decoder_channels[1])
        self.up_conv3 = ConvBlock(decoder_channels[1], decoder_channels[2])
        self.up_conv4 = ConvBlock(decoder_channels[2], decoder_channels[3])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(decoder_channels[3], n_class, kernel_size=1)

    def forward(self, encoder_feature: torch.Tensor, aux_map: torch.Tensor = None):
        if aux_map is not None:
            aux_map_resized = F.interpolate(aux_map, size=encoder_feature.shape[-2:], mode='bilinear',
                                            align_corners=False)
            bottleneck_input = torch.cat([encoder_feature, aux_map_resized], dim=1)
            x = self.bottleneck_with_aux(bottleneck_input)
        else:
            x = self.bottleneck_no_aux(encoder_feature)

        x = self.upsample(x)
        x = self.up_conv1(x)
        x = self.upsample(x)
        x = self.up_conv2(x)
        x = self.upsample(x)
        x = self.up_conv3(x)
        x = self.upsample(x)
        x = self.up_conv4(x)
        return self.final_conv(x)


class myDiceFocalLoss(_Loss):

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = True,
            sigmoid: bool = False,
            softmax: bool = True,
            other_act: Callable | None = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            gamma: float = 2.0,
            focal_weight: Sequence[float] | float | int | torch.Tensor | None = None,
            weight: Sequence[float] | float | int | torch.Tensor | None = None,
            lambda_dice: float = 1.0,
            lambda_focal: float = 1.0,
    ) -> None:
        super().__init__()
        weight = focal_weight if focal_weight is not None else weight
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction="none",
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            weight=weight,
            reduction="none",
            use_softmax=softmax
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y
        self.reduction = reduction

    def forward(
            self, input: torch.Tensor, target: torch.Tensor, pixel_weights: torch.Tensor | None = None
    ) -> torch.Tensor:

        if self.to_onehot_y and target.dim() == 3:
            target_hard = target.long()
            valid_mask = (target_hard != -1)
            target_safe = target_hard.clone()
            target_safe[~valid_mask] = 0
            target = one_hot(target_safe.unsqueeze(1), num_classes=5)
            valid_mask = valid_mask.unsqueeze(1).repeat(1, 5, 1, 1)
        elif target.dim() == 4:
            valid_mask = (target[:, 0, :, :] != -1)
            valid_mask = valid_mask.unsqueeze(1).repeat(1, 5, 1, 1)
            target_safe = target.clone()
            target_safe[~valid_mask] = 0
            target = one_hot(torch.argmax(target_safe, dim=1).unsqueeze(1), num_classes=5)
        else:
            raise ValueError(f"Target shape {target.shape} is not supported.")

        if not valid_mask.any():
            return torch.tensor(0.0, device=target.device)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        eps = 1e-8
        if pixel_weights is not None:
            if pixel_weights.shape != target.shape:
                pixel_weights = pixel_weights.unsqueeze(1).repeat(1, 5, 1, 1)
            dice_loss = (dice_loss * pixel_weights * valid_mask).sum() / (valid_mask.sum() + eps)
            focal_loss = (focal_loss * pixel_weights * valid_mask).sum() / (valid_mask.sum() + eps)
        else:
            dice_loss = (dice_loss * valid_mask).sum() / (valid_mask.sum() + eps)
            focal_loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + eps)

        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss

class SHAPE(nn.Module):

    def __init__(self, backbone, nclass=5, args: object = None):
        super().__init__()
        self.nclass = nclass
        self.embed_dim = backbone.embed_dim
        self.backbone_patch_size = backbone.patch_size
        self.args = args
        self.device = torch.device(getattr(args, 'device', 'cuda:0'))
        self.ignore_index = getattr(args, 'ignore_index', -1)
        self.ema_momentum_epoch = getattr(args, 'ema_momentum_epoch', 0.9)

        self.backbone = backbone
        self.normlayer = self.backbone.norm

        if args.overwrite_warmup_weights:
            mixup_alpha_range = getattr(args, 'mixup_alpha_range', [0.0, 1.0])
        else:
            mixup_alpha_range = getattr(args, 'mixup_alpha_range', [0.4, 0.6])

        self.use_hfm = getattr(args, 'use_hfm')
        if self.use_hfm:
            self.token_generator = MultiStrategyTokenGenerationHD(
                num_classes=nclass,
                purity_threshold=getattr(args, 'pure_tao', 0.99),
                confidence_threshold=getattr(args, 'stran_conf', 0.95),
                ignore_index=self.ignore_index,
                mixup_alpha_range=mixup_alpha_range
            )
        else:
            self.token_generator = None

        aux_channels = getattr(self.args, 'aux_channels', 1)
        self.decoder = UNetDecoder(n_class=nclass, embed_dim=self.embed_dim, aux_channels=aux_channels)
        self.ema_decoder = deepcopy(self.decoder)
        self.best_student_decoder = deepcopy(self.decoder)

        for param in self.ema_decoder.parameters():
            param.detach_()
        for param in self.best_student_decoder.parameters():
            param.detach_()

        self.criterionsup = DiceFocalLoss(softmax=True, reduction='mean')
        self.criterion = myDiceFocalLoss()

        self.use_selector_flag = getattr(args, 'use_selector', False)
        self.use_refinement_flag = getattr(args, 'use_refinement', False)
        self.use_pseudo_labels = getattr(args, 'use_pseudo_labels', False)
        if self.use_pseudo_labels:
            scorer_config = {
                "score_fusion_alpha": getattr(args, 'hpe_fusion_alpha', 0.5),
            }

            class SelectorArgs:
                def __init__(self):
                    # self.k = 0.1 
                    self.k = getattr(args, 'selector_initial_k', 0.1)

            selector_args_mock = SelectorArgs()
            anomaly_detector_params = {
                'anomaly_threshold_q': getattr(args, 'sap_threshold_percentile', 90.0)
            }
            self.pl_processor = RefiningSelector(
                num_classes=self.nclass,
                scorer_config=scorer_config,
                selector_args=selector_args_mock,
                anomaly_detector_params=anomaly_detector_params,
                use_selector_flag=self.use_selector_flag,
                use_anomaly_detector_flag=self.use_refinement_flag,
                ignore_index=-1
            )
            if self.use_refinement_flag:
                self.pl_processor.to(self.device)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._inject_capture_hooks()

    def forward(self, source_batch: Dict, target_batch: Dict, is_warmup: bool = False, current_iter: int = 0,is_sup = False):
        losses = {}
        viz_outputs = {}
        source_img, source_lbl_onehot = source_batch['s'], source_batch['label']
        img_shape = source_img.shape[-2:]
        if is_sup:
            s_feat_map, _, _ = self._extract_dino_features(source_img)
            s_logits_ori = self.decoder(s_feat_map)
            s_logits_ori_resized = F.interpolate(s_logits_ori, size=img_shape, mode='bilinear', align_corners=False)
            losses['seg_s_ori'] = self.criterionsup(s_logits_ori_resized, source_lbl_onehot)
            return viz_outputs, losses
        target_img = target_batch['s']

        B_target = target_img.shape[0]
        source_lbl = torch.argmax(source_lbl_onehot, dim=1)

        s_feat_map, _, _ = self._extract_dino_features(source_img)
        t_feat_map, _, _ = self._extract_dino_features(target_img)

        s_logits_ori = self.decoder(s_feat_map)
        s_logits_ori_resized = F.interpolate(s_logits_ori, size=img_shape, mode='bilinear', align_corners=False)
        losses['seg_s_ori'] = self.criterionsup(s_logits_ori_resized, source_lbl_onehot)

        if self.use_hfm:
            with torch.no_grad():
                if is_warmup:
                    t_logits_ema_base = self.decoder(t_feat_map)
                else:
                    t_logits_ema_base = self.ema_decoder(t_feat_map)
                t_probs_ema_base = self.robust_softmax(
                    F.interpolate(t_logits_ema_base, size=img_shape, mode='bilinear', align_corners=False))

            mixed_outputs = self.token_generator(s_feat_map, t_feat_map, source_lbl, t_probs_ema_base)

            s_feat_cross = self.normlayer(mixed_outputs["s_cross"].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            s_feat_style_to_t = self.normlayer(mixed_outputs["s_stylized_t"].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            s_logits_cross = self.decoder(s_feat_cross)
            s_logits_cross_resized = F.interpolate(s_logits_cross, size=img_shape, mode='bilinear', align_corners=False)
            losses['seg_s_cross'] = self.criterionsup(s_logits_cross_resized, source_lbl_onehot)

            s_logits_style_to_t = self.decoder(s_feat_style_to_t)
            s_logits_style_to_t_resized = F.interpolate(s_logits_style_to_t, size=img_shape, mode='bilinear',
                                                        align_corners=False)
            losses['seg_s_style_to_t'] = self.criterionsup(s_logits_style_to_t_resized, source_lbl_onehot)

            viz_outputs['s_logits_cross'] = s_logits_cross_resized.detach()

        else:
            mixed_outputs = {
                "t_stylized_s": t_feat_map,
                "t_cross": t_feat_map
            }
            losses['seg_s_cross'] = torch.tensor(0.0, device=source_img.device)
            losses['seg_s_style_to_t'] = torch.tensor(0.0, device=source_img.device)

        if not is_warmup:
            t_logits_ori_student = self.decoder(t_feat_map)
            t_logits_ori_student_resized = F.interpolate(t_logits_ori_student, size=img_shape, mode='bilinear',
                                                         align_corners=False)

            with torch.no_grad():
                t_logits_ema_base = self.ema_decoder(t_feat_map)
                pred_base = self.robust_softmax(
                    F.interpolate(t_logits_ema_base, size=img_shape, mode='bilinear', align_corners=False))

                if self.use_hfm:
                    t_feat_style_to_s = self.normlayer(mixed_outputs["t_stylized_s"].permute(0, 2, 3, 1)).permute(0, 3,
                                                                                                                  1, 2)
                    logits_style = self.ema_decoder(t_feat_style_to_s)
                    pred_style = self.robust_softmax(
                        F.interpolate(logits_style, size=img_shape, mode='bilinear', align_corners=False))

                    t_feat_cross = self.normlayer(mixed_outputs["t_cross"].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    logits_cross = self.ema_decoder(t_feat_cross)
                    pred_cross = self.robust_softmax(
                        F.interpolate(logits_cross, size=img_shape, mode='bilinear', align_corners=False))

                    teacher_preds_list = [pred_base, pred_style, pred_cross]
                else:
                    teacher_preds_list = [pred_base]

            if self.use_pseudo_labels:
                if self.use_selector_flag:
                    self.pl_processor.update_topk_ratio(current_iter)
                tensors_to_filter = (t_logits_ori_student_resized,)
                final_hard_labels, filtered_pixel_weights, filtered_tensors, selection_mask = self.pl_processor.process_labels(
                    current_iter, teacher_preds_list, *tensors_to_filter)
                filtered_student_logits = filtered_tensors[0]
                losses['pl_selected_samples'] = filtered_student_logits.shape[0]
            else:
                with torch.no_grad():
                    mean_teacher_pred = torch.mean(torch.stack(teacher_preds_list), dim=0)
                    final_hard_labels = torch.argmax(mean_teacher_pred, dim=1)
                    filtered_student_logits = t_logits_ori_student_resized
                    filtered_pixel_weights = None
                    selection_mask = torch.ones(B_target, dtype=torch.bool, device=target_img.device)

            if final_hard_labels.numel() > 0:
                losses['pseudo_t_ori'] = self.criterion(
                    filtered_student_logits, final_hard_labels, pixel_weights=filtered_pixel_weights)
            else:
                losses['pseudo_t_ori'] = torch.tensor(0.0, device=target_img.device)

        viz_outputs['source_img'] = source_img.detach()
        viz_outputs['source_lbl'] = source_lbl.detach()
        if is_warmup:
            viz_outputs['s_logits_ori'] = s_logits_ori_resized.detach()

        if not is_warmup and torch.any(selection_mask):
            viz_outputs['target_img'] = target_img[selection_mask].detach()
            viz_outputs['target_logits_ori_student'] = filtered_student_logits.detach()
            viz_outputs['pseudo_labels'] = final_hard_labels.detach()
            viz_outputs['target_logits_ema'] = pred_base[selection_mask].detach()

        return viz_outputs, losses

    def inference(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feature_map, _, deepest_attn_map = self._extract_dino_features(image)
            logits = self.decoder(feature_map)
            return F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)

    def load_warmup_weights(self, path: str):
        try:
            warmup_state_dict = torch.load(path, map_location=self.device)
            missing_keys, unexpected_keys = self.decoder.load_state_dict(
                warmup_state_dict, strict=True
            )
            if missing_keys:
                print(f"WARNING: The following keys were missing in the state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"WARNING: The following keys were unexpected in the state_dict: {unexpected_keys}")

            if not missing_keys and not unexpected_keys:
                print("Student decoder weights loaded successfully with NO mismatch.")
            else:
                print("CRITICAL: State dict mismatch detected. This can lead to NaN/Inf outputs.")

            self.ema_decoder = deepcopy(self.decoder)
            self.best_student_decoder = deepcopy(self.decoder)

            self.decoder.train()
            self.ema_decoder.eval()
            self.best_student_decoder.eval()

        except FileNotFoundError:
            print(f"Error: Warm-up weights file not found at {path}. Cannot start UDA phase.")
            raise
        except Exception as e:
            print(f"An error occurred while loading weights: {e}")
            import traceback
            traceback.print_exc()
            raise

    def on_epoch_start(self):
        if self.use_pseudo_labels and hasattr(self, 'pl_processor') and self.use_selector_flag:
            self.pl_processor.reset_epoch()

    def _inject_capture_hooks(self):
        for block in self.backbone.blocks:
            block.attn.compute_attention = types.MethodType(get_capture_attention_fn(block.attn), block.attn)


    @torch.no_grad()
    def _ema_update(self, student_model, teacher_model, alpha=0.999):
        student_state_dict = student_model.state_dict()
        for key, teacher_param in teacher_model.state_dict().items():
            student_param = student_state_dict[key]
            if teacher_param.dtype.is_floating_point:
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
            else:
                teacher_param.data.copy_(student_param.data)

    @torch.no_grad()
    def _update_ema_batch(self, alpha=0.999):
        self._ema_update(
            student_model=self.decoder,
            teacher_model=self.ema_decoder,
            alpha=alpha
        )

    def _extract_dino_features(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            B, _, img_h, img_w = x.shape
            patch_h, patch_w = img_h // self.backbone_patch_size, img_w // self.backbone_patch_size
            num_patch_tokens = patch_h * patch_w
            intermediate_features_raw = self.backbone.get_intermediate_layers(x, n=[11], norm=True, reshape=False)[0]
            raw_attn_map = features_storage.get(id(self.backbone.blocks[11].attn), {}).get('attn_map')

            if raw_attn_map is None:
                if intermediate_features_raw.shape[1] == num_patch_tokens:
                    feature_map = intermediate_features_raw.permute(0, 2, 1).reshape(B, self.embed_dim, patch_h,
                                                                                     patch_w)
                    return feature_map, None, None
                else:
                    raise ValueError("Cannot extract features: attn_map is missing and feature shape is ambiguous.")

            num_total_tokens = raw_attn_map.shape[-1]
            n_extra_tokens = num_total_tokens - num_patch_tokens

            if intermediate_features_raw.shape[1] == num_patch_tokens:
                patch_features = intermediate_features_raw
            elif intermediate_features_raw.shape[1] == num_total_tokens:
                patch_features = intermediate_features_raw[:, n_extra_tokens:]
            else:
                raise ValueError(
                    f"Feature map has an unexpected number of tokens! "
                    f"Expected {num_patch_tokens} or {num_total_tokens}, but got {intermediate_features_raw.shape[1]}."
                )

            feature_map = patch_features.permute(0, 2, 1).reshape(B, self.embed_dim, patch_h, patch_w)

            attn_heads_avg = raw_attn_map.mean(dim=1)
            cls_to_patches_attn = attn_heads_avg[:, 0, n_extra_tokens:]

            if cls_to_patches_attn.shape[1] != num_patch_tokens:
                raise ValueError("Shape mismatch after slicing attention map!")

            processed_attn_map = cls_to_patches_attn.reshape(B, 1, patch_h, patch_w)

            p_min = processed_attn_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            p_max = processed_attn_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            processed_attn_map = (processed_attn_map - p_min) / (p_max - p_min + 1e-8)

        return feature_map, raw_attn_map, processed_attn_map

    def robust_softmax(self, logits, temperature=1.0, clamp_range=(-100.0, 100.0)):
        if not torch.isfinite(logits).all():
            print("WARNING: Non-finite values (NaN/Inf) detected in logits. Clamping.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_range[1], neginf=clamp_range[0])
        clamped_logits = torch.clamp(logits, min=clamp_range[0], max=clamp_range[1])
        return F.softmax(clamped_logits / temperature, dim=1)
