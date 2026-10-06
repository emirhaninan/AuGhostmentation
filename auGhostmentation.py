import math
import random
import numpy as np
from scipy.stats import lognorm as lognorm_dist
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuGhostmentation(nn.Module):

    _LN_SIGMA = 0.53
    _MEDIAN_ARCMIN = 19.4

    def __init__(self,
                 fov_deg=8.0,
                 image_size=224,
                 expected_ms=1.33,
                 ms_horizontal_weight=0.7193,
                 ms_cardinal_kappa=7.5347,
                 w=0.50,
                 min_arcmin=5.0,
                 max_arcmin=60.0):
        super().__init__()

        ppd = image_size / fov_deg
        arcmin_per_px = 60.0 / ppd

        self.min_shift = min_arcmin / arcmin_per_px
        self.max_shift = max_arcmin / arcmin_per_px

        self._ln_scale = self._MEDIAN_ARCMIN / arcmin_per_px
        self._cdf_lo = lognorm_dist.cdf(
            self.min_shift, self._LN_SIGMA, scale=self._ln_scale)
        self._cdf_hi = lognorm_dist.cdf(
            self.max_shift, self._LN_SIGMA, scale=self._ln_scale)

        self.expected_ms = expected_ms
        self.ms_horizontal_weight = ms_horizontal_weight
        self.ms_cardinal_kappa = ms_cardinal_kappa
        self.w = w

    def _sample_direction(self):
        if random.random() < self.ms_horizontal_weight:
            center = random.choice([0.0, math.pi])
        else:
            center = random.choice([math.pi / 2, 3 * math.pi / 2])

        angle = float(np.random.vonmises(center, self.ms_cardinal_kappa))
        dx = math.cos(angle)
        dy = math.sin(angle)
        mag = math.sqrt(dx ** 2 + dy ** 2) + 1e-8
        return dx / mag, dy / mag

    def _sample_amplitude(self):
        u = np.random.uniform(self._cdf_lo, self._cdf_hi)
        return float(lognorm_dist.ppf(
            u, self._LN_SIGMA, scale=self._ln_scale))

    def forward(self, img):
        n_ms = np.random.poisson(self.expected_ms)
        if n_ms == 0:
            return img

        result = img
        for _ in range(n_ms):
            dir_x, dir_y = self._sample_direction()
            amplitude = self._sample_amplitude()

            dx = amplitude * dir_x
            dy = amplitude * dir_y

            K = max(2, int(round(amplitude)))
            ks = (K * 2) + 1
            kernel = torch.zeros(ks, ks, device=result.device,
                                 dtype=result.dtype)
            cx = K
            cy = K
            num_samples = K * 2
            for k in range(num_samples):
                t = k / (num_samples - 1)
                px = int(round(cx + t * dx))
                py = int(round(cy + t * dy))
                px = max(0, min(ks - 1, px))
                py = max(0, min(ks - 1, py))
                kernel[py, px] += 1.0
            kernel = kernel / (kernel.sum() + 1e-8)

            C = result.shape[0]
            k2d = kernel.view(1, 1, ks, ks).expand(C, 1, ks, ks)
            pad = ks // 2
            blurred = F.conv2d(result.unsqueeze(0), k2d,
                               padding=pad, groups=C).squeeze(0)

            result = result * (1.0 - self.w) + blurred * self.w

        return result


tiny_imagenet_aughost = AuGhostmentation(
    fov_deg=8.0,
    image_size=64,
    expected_ms=1.33,
    w=0.50,
)

imagenet_aughost = AuGhostmentation(
    fov_deg=8.0,
    image_size=224,
    expected_ms=0.33,
    w=0.50,
)
