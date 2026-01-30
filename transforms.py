import torch
import random
from typing import List, Tuple, Dict, Any, Union
from torchvision import transforms as T
from torchvision.transforms import functional as F


class BaseDictTransform:
    def __init__(self, keys: List[str], prob: float = 1.0, allow_missing_keys: bool = True):
        self.keys = keys
        self.prob = prob
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.prob:
            return sample

        actual_keys = self.get_actual_keys(sample)
        if not actual_keys:
            return sample

        return self._apply(sample, actual_keys)

    def get_actual_keys(self, sample: Dict[str, Any]) -> List[str]:
        present_keys = [key for key in self.keys if key in sample and sample[key] is not None]
        if not present_keys and not self.allow_missing_keys:
            raise KeyError(f"None of the keys {self.keys} are present in the sample.")
        return present_keys

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


class RandAffined(BaseDictTransform):

    def __init__(self, keys: List[str], prob: float = 1.0,
                 rotate_range: Union[float, Tuple[float, float]] = 0,
                 scale_range: Union[float, Tuple[float, float]] = 0,
                 translate_range: Union[float, Tuple[float, float]] = 0,
                 allow_missing_keys: bool = True):
        super().__init__(keys, prob, allow_missing_keys)

        self.degrees = (-rotate_range, rotate_range) if isinstance(rotate_range, (int, float)) else rotate_range
        self.scale = (1.0 - scale_range, 1.0 + scale_range) if isinstance(scale_range, (int, float)) else scale_range
        self.translate = (translate_range, translate_range) if isinstance(translate_range,
                                                                          (int, float)) else translate_range

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        img_for_size = sample[actual_keys[0]]
        params = T.RandomAffine.get_params(self.degrees, self.translate, self.scale, None, img_for_size.shape[-2:])
        angle, translations, scale, shear = params
        for key in actual_keys:
            img = sample[key]
            is_mask = (key == 'label')
            interpolation = T.InterpolationMode.NEAREST if is_mask else T.InterpolationMode.BILINEAR
            sample[key] = F.affine(img, angle, list(translations), scale, [0.0, 0.0], interpolation=interpolation)

        return sample


class RandZoomd(RandAffined):
    def __init__(self, keys: List[str], prob: float = 1.0,
                 min_zoom: float = 1.0, max_zoom: float = 1.0,
                 keep_size: bool = True, allow_missing_keys: bool = True):
        if not keep_size:
            raise NotImplementedError("Only keep_size=True is implemented for RandZoomd.")

        scale_range = (min_zoom, max_zoom)
        super().__init__(keys, prob, scale_range=scale_range, allow_missing_keys=allow_missing_keys)


class Resized(BaseDictTransform):
    def __init__(self, keys: List[str], spatial_size: Tuple[int, int], allow_missing_keys: bool = True):
        super().__init__(keys, prob=1.0, allow_missing_keys=allow_missing_keys)
        self.spatial_size = spatial_size

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        for key in actual_keys:
            is_mask = (key == 'label')
            interpolation = T.InterpolationMode.NEAREST if is_mask else T.InterpolationMode.BILINEAR
            sample[key] = F.resize(sample[key], list(self.spatial_size), interpolation=interpolation)
        return sample


class RepeatChanneld(BaseDictTransform):
    def __init__(self, keys: List[str], repeats: int = 3, allow_missing_keys: bool = True):
        super().__init__(keys, prob=1.0, allow_missing_keys=allow_missing_keys)
        self.repeats = repeats

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        for key in actual_keys:
            img = sample[key]
            if img.shape[0] == 1:
                sample[key] = img.repeat(self.repeats, 1, 1)
        return sample


class ScaleTo01(BaseDictTransform):

    def __init__(self, keys: List[str]):
        super().__init__(keys, prob=1.0)

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        for key in actual_keys:
            if key != 'label':
                img = sample[key]
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    sample[key] = (img - min_val) / (max_val - min_val)
                else:
                    sample[key] = torch.zeros_like(img)
        return sample


class RandAdjustContrastd(BaseDictTransform):
    def __init__(self, keys: List[str], prob: float = 1.0,
                 gamma: Tuple[float, float] = (0.5, 4.5),
                 allow_missing_keys: bool = True):
        super().__init__(keys, prob, allow_missing_keys)
        self.gamma_range = gamma

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])

        for key in actual_keys:
            if key != 'label':
                img = sample[key]
                if img.min() < 0:
                    print(f"WARNING: Input to RandAdjustContrastd for key '{key}' has negative values. "
                          f"Min: {img.min()}. Skipping contrast adjustment for this sample.")
                    continue

                sample[key] = F.adjust_gamma(img, gamma)
        return sample


class NormalizeIntensityd(BaseDictTransform):
    def __init__(self, keys: List[str], subtrahend: List[float], divisor: List[float], allow_missing_keys: bool = True):
        super().__init__(keys, prob=1.0, allow_missing_keys=allow_missing_keys)
        std_tensor = torch.tensor(divisor)
        self.normalize = T.Normalize(mean=subtrahend, std=(std_tensor + 1e-8).tolist())

    def _apply(self, sample: Dict[str, Any], actual_keys: List[str]) -> Dict[str, Any]:
        for key in actual_keys:
            if key != 'label':
                sample[key] = self.normalize(sample[key])
        return sample
