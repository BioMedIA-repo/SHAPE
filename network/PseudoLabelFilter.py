import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Any, Dict
from collections import deque
import itertools
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

try:
    from skimage.measure import label as skimage_label
    from joblib import Parallel, delayed

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image or joblib not found. LCC filtering will be disabled.")


def sigmoid_rampup(current_iter, rampup_length):
    if rampup_length == 0:
        return 1.0
    if current_iter >= rampup_length:
        return 1.0

    current = max(0.0, float(current_iter))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


@torch.jit.script
def _get_smoothed_shape_properties_jit(
        seg_map_one_hot: Tensor,
        erosion_kernel: Tensor,
        dilation_kernel: Tensor
) -> Tensor:
    seg_map_one_hot = seg_map_one_hot.float()

    kernel_h, kernel_w = erosion_kernel.shape
    kernel_size = (kernel_h, kernel_w)
    padding = (kernel_h - 1) // 2

    eroded_mask = -F.max_pool2d(-seg_map_one_hot, kernel_size=kernel_size, stride=1, padding=padding)
    opened_mask = F.max_pool2d(eroded_mask, kernel_size=kernel_size, stride=1, padding=padding)

    dilated_mask = F.max_pool2d(opened_mask, kernel_size=kernel_size, stride=1, padding=padding)
    smoothed_mask = -F.max_pool2d(-dilated_mask, kernel_size=kernel_size, stride=1, padding=padding)

    areas = torch.sum(smoothed_mask, dim=(2, 3))

    dilated_for_perimeter = F.max_pool2d(smoothed_mask, kernel_size=3, stride=1, padding=1)
    perimeters = torch.sum(dilated_for_perimeter - smoothed_mask, dim=(2, 3))

    isoperimetric_ratio = (4 * torch.pi * areas) / (perimeters.pow(2) + 1e-8)
    isoperimetric_ratio.nan_to_num_(0.0)

    return isoperimetric_ratio.clamp_(0.0, 1.0)


@torch.jit.script
def _get_class_bboxes_jit(seg_map: Tensor, n_class: int) -> Tuple[Tensor, Tensor]:
    B, H, W = seg_map.shape
    device = seg_map.device

    one_hot = F.one_hot(seg_map, n_class).permute(0, 3, 1, 2)
    has_class = torch.any(one_hot.flatten(start_dim=2), dim=2)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    x_coords = grid_x.view(1, 1, H, W)
    y_coords = grid_y.view(1, 1, H, W)
    inf_val = float('inf')
    neg_inf_val = float('-inf')

    x_min = torch.where(one_hot > 0, x_coords, inf_val).flatten(2).min(2).values
    x_max = torch.where(one_hot > 0, x_coords, neg_inf_val).flatten(2).max(2).values
    y_min = torch.where(one_hot > 0, y_coords, inf_val).flatten(2).min(2).values
    y_max = torch.where(one_hot > 0, y_coords, neg_inf_val).flatten(2).max(2).values

    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=2)

    bboxes[torch.isinf(bboxes)] = -1.0

    return bboxes, has_class


@torch.jit.script
def _get_relative_direction_jit(bboxes: Tensor, has_class: Tensor) -> Tuple[Tensor, Tensor]:
    B, C, _ = bboxes.shape
    device = bboxes.device
    eps = 1e-8

    bbox_centers_all = torch.stack([
        (bboxes[..., 0] + bboxes[..., 2]) / 2.0,
        (bboxes[..., 1] + bboxes[..., 3]) / 2.0
    ], dim=-1)

    has_class_expanded = has_class.unsqueeze(-1).float()
    sum_centers = torch.sum(bbox_centers_all * has_class_expanded, dim=1)
    num_present_classes = has_class.sum(dim=1, keepdim=True).float()
    fg_centers = sum_centers / (num_present_classes + eps)

    centers1 = bbox_centers_all.unsqueeze(2)
    centers2 = bbox_centers_all.unsqueeze(1)

    vec_c1_c2 = centers1 - centers2
    vec_c1_fg = fg_centers.view(-1, 1, 1, 2) - centers1

    dot_product = (vec_c1_c2 * vec_c1_fg).sum(dim=-1)
    norm_prod = torch.norm(vec_c1_c2, dim=-1) * torch.norm(vec_c1_fg, dim=-1)

    relative_angle_cos = dot_product / (norm_prod + eps)
    relative_angle_cos.nan_to_num_(0.0)

    has_pair = has_class.unsqueeze(2) & has_class.unsqueeze(1)

    return relative_angle_cos, has_pair


class StructuralScorer(nn.Module):

    def __init__(self, n_class: int, score_fusion_alpha: float = 0.5, softmin_tau: float = 0.1,
                 history_size: int = 1024):
        super().__init__()
        self.n_class = n_class
        self.score_fusion_alpha = score_fusion_alpha
        self.softmin_tau = softmin_tau
        self.history_size = history_size
        self.register_buffer('morph_kernel', torch.ones(5, 5))

        self.shape_queues = {c: deque(maxlen=history_size) for c in range(n_class)}

        self.layout_pairs = list(itertools.product(range(n_class), range(n_class)))
        self.layout_queues = {pair: deque(maxlen=history_size) for pair in self.layout_pairs if pair[0] != pair[1]}

        print(f"Initialized StructuralScorer (Queue-Based, History={history_size}, Soft-Min tau={softmin_tau}).")

    def _update_histories(self, batch_shape_ratios: Tensor, batch_directions: Tensor,
                          has_class_mask: Tensor, has_pair_mask: Tensor):

        B, C = batch_shape_ratios.shape

        shape_data = batch_shape_ratios.detach().cpu()
        mask_data = has_class_mask.detach().cpu()

        for c in range(C):
            valid_indices = torch.where(mask_data[:, c])[0]
            if len(valid_indices) > 0:
                values = shape_data[valid_indices, c].tolist()
                self.shape_queues[c].extend(values)

        direction_data = batch_directions.detach().cpu()
        pair_mask_data = has_pair_mask.detach().cpu()

        for (c1, c2) in self.layout_queues.keys():
            valid_indices = torch.where(pair_mask_data[:, c1, c2])[0]
            if len(valid_indices) > 0:
                values = direction_data[valid_indices, c1, c2].tolist()
                self.layout_queues[(c1, c2)].extend(values)

    def _get_stats_from_queue(self, device):
        shape_means = torch.zeros(self.n_class, device=device)
        shape_stds = torch.ones(self.n_class, device=device)  # 默认为1，避免除以0

        for c in range(self.n_class):
            if len(self.shape_queues[c]) > 1:
                data = torch.tensor(list(self.shape_queues[c]), device=device, dtype=torch.float32)
                shape_means[c] = data.mean()
                shape_stds[c] = data.std() + 1e-6
            elif len(self.shape_queues[c]) == 1:
                shape_means[c] = torch.tensor(list(self.shape_queues[c])[0], device=device)
                shape_stds[c] = 0.1

        layout_means = torch.zeros(self.n_class, self.n_class, device=device)
        layout_stds = torch.ones(self.n_class, self.n_class, device=device)

        for (c1, c2), queue in self.layout_queues.items():
            if len(queue) > 1:
                data = torch.tensor(list(queue), device=device, dtype=torch.float32)
                layout_means[c1, c2] = data.mean()
                layout_stds[c1, c2] = data.std() + 1e-6
            elif len(queue) == 1:
                layout_means[c1, c2] = torch.tensor(list(queue)[0], device=device)
                layout_stds[c1, c2] = 0.1

        return shape_means, shape_stds, layout_means, layout_stds

    @staticmethod
    def _softmin(x: Tensor, mask: Tensor, tau: float) -> Tensor:
        x_masked = torch.where(mask, x, torch.finfo(x.dtype).max)
        weights = F.softmax(-x_masked / tau, dim=1)
        soft_min_val = torch.sum(x * weights, dim=1)
        any_valid = mask.any(dim=1)
        return torch.where(any_valid, soft_min_val, 1.0)

    @torch.no_grad()
    def get_scores(self, pred_map_batch: Tensor) -> Tensor:
        B = pred_map_batch.shape[0]
        one_hot = F.one_hot(pred_map_batch, self.n_class).permute(0, 3, 1, 2)
        target_shape_ratios = _get_smoothed_shape_properties_jit(one_hot, self.morph_kernel, self.morph_kernel)
        bboxes, has_class = _get_class_bboxes_jit(pred_map_batch, self.n_class)
        directions, has_pair = _get_relative_direction_jit(bboxes, has_class)
        has_class_mask = torch.any(one_hot.flatten(2), dim=2)
        self._update_histories(target_shape_ratios, directions, has_class_mask, has_pair)
        hist_shape_mean, hist_shape_std, hist_layout_mean, hist_layout_std = \
            self._get_stats_from_queue(pred_map_batch.device)
        shape_dev_z = torch.abs(target_shape_ratios - hist_shape_mean.unsqueeze(0)) / hist_shape_std.unsqueeze(0)
        shape_sim = torch.exp(-shape_dev_z)
        layout_dev_z = torch.abs(directions - hist_layout_mean.unsqueeze(0)) / hist_layout_std.unsqueeze(0)
        layout_sim = torch.exp(-layout_dev_z)
        intra_scores = self._softmin(shape_sim, has_class_mask, self.softmin_tau)
        inter_scores = self._softmin(layout_sim.flatten(start_dim=1), has_pair.flatten(start_dim=1), self.softmin_tau)
        final_scores = ((1 - self.score_fusion_alpha) * intra_scores +
                        self.score_fusion_alpha * inter_scores)
        return final_scores.detach()

    def calibrate(self, *args, **kwargs):
        pass


class EpochProxySelector:

    def __init__(self, num_classes, hypergraph_scorer: StructuralScorer, args=None, history_len=1024):
        self.score_queue = deque(maxlen=history_len)
        self.topK_ratio = 0.0
        self.num_classes = num_classes
        self.log_c = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
        self.args = args
        self.score_threshold = 0.3
        self.hypergraph_scorer = hypergraph_scorer
        self.w_structure = getattr(args, 'w_structure', 1.0)
        self.DISCARD_SCORE = -1.0
        self.BACKGROUND_IDX = 0
        print(f"Initialized EpochProxySelector (Queue-Based, MaxLen={history_len})")

    def reset_epoch(self):
        pass

    def update_topk_ratio(self, iter_num, rampup_length=40000):
        start_k = self.args.k if self.args and hasattr(self.args, 'k') else 0.1
        ramp = sigmoid_rampup(iter_num, rampup_length)
        self.topK_ratio = start_k + (1.0 - start_k) * ramp
        self.topK_ratio = min(self.topK_ratio, 1.0)

    def _compute_pixel_weights(self, predictions_list: List[torch.Tensor]) -> Tensor:
        assert torch.allclose(torch.sum(predictions_list[0], dim=1),
                              torch.tensor(1.0, device=predictions_list[0].device)), \
            "Input to _compute_pixel_weights must be softmax probabilities."
        device = predictions_list[0].device
        self.log_c = self.log_c.to(device)
        epsilon = 1e-6
        if len(predictions_list) < 2:
            single_pred = predictions_list[0]
            entropy = -torch.sum(single_pred * torch.log(single_pred + epsilon), dim=1)
            pixel_certainty = 1 - (entropy / self.log_c)
            return pixel_certainty.detach()

        stacked_preds = torch.stack(predictions_list, dim=1)
        mean_pred = torch.mean(stacked_preds, dim=1)
        entropy_of_mean = -torch.sum(mean_pred * torch.log(mean_pred + epsilon), dim=1)
        entropies = -torch.sum(stacked_preds * torch.log(stacked_preds + epsilon), dim=2)
        mean_of_entropies = torch.mean(entropies, dim=1)
        jsd_pixels = entropy_of_mean - mean_of_entropies
        pixel_consistency = 1 - (jsd_pixels / self.log_c)
        pixel_certainty = 1 - (entropy_of_mean / self.log_c)
        pixel_weights = (pixel_consistency * pixel_certainty).detach()
        del mean_pred, entropy_of_mean, mean_of_entropies, pixel_consistency, pixel_certainty
        return pixel_weights

    def _compute_sample_scores(self, predictions_list: List[torch.Tensor]) -> Tuple[Tensor, Tensor]:
        pixel_weights = self._compute_pixel_weights(predictions_list)

        if len(predictions_list) > 1:
            mean_pred = torch.mean(torch.stack(predictions_list, dim=0), dim=0)
        else:
            mean_pred = predictions_list[0]
        hard_pred_from_consensus = torch.argmax(mean_pred, dim=1)
        foreground_mask = (hard_pred_from_consensus != self.BACKGROUND_IDX).float()
        foreground_pixel_weights = pixel_weights * foreground_mask
        sum_foreground_weights = torch.sum(foreground_pixel_weights, dim=(1, 2))
        num_foreground_pixels = torch.sum(foreground_mask, dim=(1, 2))
        pixel_sample_scores = sum_foreground_weights / (num_foreground_pixels + 1e-8)
        structural_score = self.hypergraph_scorer.get_scores(hard_pred_from_consensus)
        if self.w_structure > 0:
            gating_factor = structural_score.pow(self.w_structure)
            final_sample_scores = pixel_sample_scores * gating_factor
        else:
            final_sample_scores = pixel_sample_scores
        has_foreground = torch.any((hard_pred_from_consensus > 0).flatten(start_dim=1), dim=1)
        final_sample_scores[~has_foreground] = self.DISCARD_SCORE

        return final_sample_scores, pixel_weights

    def update_score_distribution(self, predictions_list: List[torch.Tensor]):
        sample_scores, _ = self._compute_sample_scores(predictions_list)
        valid_scores = [s for s in sample_scores.cpu().tolist() if s >= 0]
        self.score_queue.extend(valid_scores)

    def update_selection_threshold(self):
        if not self.score_queue:
            return
        all_scores = np.array(self.score_queue)
        target_percentile = (1.0 - self.topK_ratio) * 100
        self.score_threshold = np.percentile(all_scores, target_percentile)

    def filter_batch(
            self,
            predictions_list: List[torch.Tensor],
            *other_tensors: Tensor
    ):
        device = predictions_list[0].device
        batch_size = predictions_list[0].size(0)
        sample_scores, pixel_weights = self._compute_sample_scores(predictions_list)
        selected_mask = (sample_scores >= self.score_threshold) & (sample_scores >= 0)
        num_selected = selected_mask.sum()
        expected_num = max(1, int(batch_size * self.topK_ratio))
        final_selection_mask = selected_mask

        if num_selected < expected_num:
            final_selection_mask = selected_mask.clone()
            num_needed = expected_num - num_selected
            remaining_mask = ~selected_mask
            remaining_scores = sample_scores[remaining_mask]
            num_to_select = min(num_needed, remaining_scores.size(0))
            if num_to_select > 0:
                _, topk_indices_in_remaining = torch.topk(remaining_scores, k=num_to_select)
                original_indices_of_remaining = torch.where(remaining_mask)[0]
                selected_original_indices = original_indices_of_remaining[topk_indices_in_remaining]
                final_selection_mask[selected_original_indices] = True

        if not final_selection_mask.any():
            C, H, W = predictions_list[0].shape[1:]
            empty_pred = torch.empty(0, C, H, W, device=device)
            empty_weights = torch.empty(0, H, W, device=device)
            empty_others = tuple(torch.empty(0, *t.shape[1:], device=device) for t in other_tensors)
            return empty_pred, empty_weights, empty_others, final_selection_mask

        final_preds_list = [p[final_selection_mask] for p in predictions_list]
        final_pixel_weights = pixel_weights[final_selection_mask]
        final_other_tensors = tuple(t[final_selection_mask] for t in other_tensors)

        final_pred = torch.mean(torch.stack(final_preds_list, dim=0), dim=0)
        return final_pred, final_pixel_weights, final_other_tensors, final_selection_mask


class CrossViewAnomalyDetector(nn.Module):
    def __init__(self, num_classes: int, anomaly_threshold_q: float = 85.0, min_area_ratio: float = 0.01,
                 ignore_index: int = -1, exclude_background: bool = True, use_lcc_filtering: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.anomaly_threshold_q = anomaly_threshold_q
        self.min_area_ratio = min_area_ratio
        self.ignore_index = ignore_index
        self.exclude_background_idx = 0 if exclude_background else -1
        self.n_jobs = -1

    @staticmethod
    def _process_single_image_lcc_skimage(image_np: np.ndarray, num_classes: int, ignore_index: int,
                                          bg_idx: int) -> np.ndarray:
        final_image = np.full(image_np.shape, ignore_index, dtype=image_np.dtype)
        for c in range(num_classes):
            if c == bg_idx: final_image[image_np == c] = c; continue
            class_mask = (image_np == c)
            if not class_mask.any(): continue
            labeled_mask, num_labels = skimage_label(class_mask, background=0, connectivity=1, return_num=True)
            if num_labels == 0: continue
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_component_label = np.argmax(component_sizes) + 1
            lcc_mask = (labeled_mask == largest_component_label)
            final_image[lcc_mask] = c
        return final_image

    @torch.no_grad()
    def forward(self, predictions_list: List[torch.Tensor], pseudo_labels_to_modify: torch.Tensor,
                current_iter=0) -> torch.Tensor:
        if not predictions_list or pseudo_labels_to_modify.numel() == 0:
            return pseudo_labels_to_modify
        B, C, H, W = predictions_list[0].shape
        device = predictions_list[0].device
        total_pixels = H * W
        hard_preds = [torch.argmax(p, dim=1) for p in predictions_list]

        pixel_counts_per_view = []
        for p in hard_preds:
            one_hot_p = F.one_hot(p, num_classes=5).permute(0, 3, 1, 2)
            counts = torch.sum(one_hot_p, dim=(2, 3))
            pixel_counts_per_view.append(counts)

        stacked_counts = torch.stack(pixel_counts_per_view, dim=2)
        mean_counts = torch.mean(stacked_counts.float(), dim=2)
        std_counts = torch.std(stacked_counts.float(), dim=2)
        anomaly_scores = std_counts / (mean_counts + 1e-8)
        anomaly_scores[mean_counts == 0] = 0.0
        significant_class_mask = (mean_counts / total_pixels) > self.min_area_ratio
        if self.exclude_background_idx != -1:
            significant_class_mask[:, self.exclude_background_idx] = False
        scores_to_consider = anomaly_scores[significant_class_mask]
        if scores_to_consider.numel() > 0:
            anomaly_threshold = torch.quantile(scores_to_consider, self.anomaly_threshold_q / 100.0)
        else:
            anomaly_threshold = float('inf')
        is_anomalous_class = (anomaly_scores > anomaly_threshold) & significant_class_mask
        final_labels = pseudo_labels_to_modify.clone()
        valid_pixel_mask = (final_labels != self.ignore_index)

        for b in range(B):
            if not valid_pixel_mask[b].any():
                continue

            anomalous_indices = torch.where(is_anomalous_class[b])[0]

            if anomalous_indices.numel() > 0:
                pixel_mask = torch.isin(final_labels[b], anomalous_indices)
                final_labels[b][pixel_mask] = self.ignore_index
        return final_labels


class RefiningSelector:

    def __init__(self, num_classes, scorer_config: Dict[str, Any], selector_args=None,
                 anomaly_detector_params: Dict[str, Any] = None, use_selector_flag=True,
                 use_anomaly_detector_flag=True, ignore_index=-1):
        print("Initializing PseudoLabelProcessor...")
        self.use_selector_flag = use_selector_flag
        self.use_anomaly_detector_flag = use_anomaly_detector_flag
        self.ignore_index = ignore_index
        if self.use_selector_flag:
            print("Using StructuralScorer for hypergraph scoring.")
            self.hypergraph_scorer = StructuralScorer(n_class=num_classes, **scorer_config)
            self.selector = EpochProxySelector(num_classes, args=selector_args,
                                               hypergraph_scorer=self.hypergraph_scorer)
        if self.use_anomaly_detector_flag:
            print("Using CrossViewAnomalyDetector for anomaly detection.")
            if anomaly_detector_params is None: anomaly_detector_params = {}
            self.anomaly_detector = CrossViewAnomalyDetector(num_classes=num_classes, ignore_index=self.ignore_index,
                                                             **anomaly_detector_params)

    def to(self, device):
        if hasattr(self, 'anomaly_detector'): self.anomaly_detector.to(device)
        return self

    def reset_epoch(self):
        self.selector.reset_epoch()

    def update_topk_ratio(self, iter_num, rampup_length=40000):
        self.selector.update_topk_ratio(iter_num, rampup_length)

    def calibrate(self, source_loader: DataLoader, device: torch.device):
        self.hypergraph_scorer.calibrate(source_loader, device)

    def process_labels(self, current_iter=None, predictions_list: List[torch.Tensor] = None, *other_tensors):
        device = predictions_list[0].device
        batch_size = predictions_list[0].size(0)
        with torch.no_grad():
            full_batch_soft_pred = torch.mean(torch.stack(predictions_list, dim=0), dim=0)

        pixel_weights = None
        tensors_to_return = other_tensors
        selection_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        if self.use_selector_flag:
            self.selector.update_score_distribution(predictions_list)
            self.selector.update_selection_threshold()
            filter_batch_outputs = self.selector.filter_batch(predictions_list, *other_tensors)
            _, pixel_weights, _, selection_mask = filter_batch_outputs
            if not selection_mask.any():
                H, W = full_batch_soft_pred.shape[2:]
                empty_hard_labels = torch.empty(0, H, W, dtype=torch.long, device=device)
                empty_others = tuple(torch.empty(0, *t.shape[1:], device=device) for t in other_tensors)
                return empty_hard_labels, pixel_weights, empty_others, selection_mask
        hard_pseudo_labels_full_batch = torch.argmax(full_batch_soft_pred, dim=1)
        if self.use_anomaly_detector_flag and current_iter <= 30000:
            hard_pseudo_labels_full_batch = self.anomaly_detector(
                predictions_list,
                hard_pseudo_labels_full_batch,
                current_iter
            )
        final_hard_labels = hard_pseudo_labels_full_batch[selection_mask]
        tensors_to_return = tuple(t[selection_mask] for t in other_tensors)
        return final_hard_labels, pixel_weights, tensors_to_return, selection_mask
