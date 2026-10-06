# AuGhostmentation

Code for "AuGhostmentation: The Eyes Never Stand Still—Why Should Neural Networks?" (Emirhan Inan and Suayb S. Arslan, NeurIPS 2026 [Main Track | Poster]).

AuGhostmentation is a training-time image augmentation based on fixational eye movements. For each image, it draws a Poisson number of microsaccades and smears the image alongside a directional motion kernel, blending the result with the current frame. Kernel directions follow the cardinal bias and kernel lengths follow the amplitude distribution measured from human eye-tracking data (GazeBase v2.0). The only parameter that depends on the dataset is the expected number of microsaccades per image, `expected_ms`, which also denotes gaze duration.

## Requirements

- Python
- PyTorch
- NumPy
- SciPy

## Usage

`AuGhostmentation` is a `torch.nn.Module` that takes a single image tensor of shape `(C, H, W)` with values in `[0, 1]` and returns a tensor of the same shape. Apply it after `ToTensor()` and before normalization, during training only.

```python
from torchvision import transforms
from aughostmentation import AuGhostmentation

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    AuGhostmentation(image_size=224, expected_ms=0.33),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

The two settings used in the paper are included:

```python
from aughostmentation import tiny_imagenet_aug, imagenet_aug

tiny_imagenet_aug  # Tiny ImageNet, 64 x 64 px, expected_ms = 1.33
imagenet_aug       # ImageNet-100, 224 x 224 px, expected_ms = 0.33
```

## Parameters

| Argument | Default | Description |
|---|---|---|
| `fov_deg` | 8.0 | Field of view covered by the image, in degrees |
| `image_size` | 224 | Image side length, in pixels |
| `expected_ms` | 1.33 | Expected number of microsaccades per image (Poisson rate λ) |
| `ms_horizontal_weight` | 0.7193 | Probability that a microsaccade is drawn around the horizontal axis (p_h) |
| `ms_cardinal_kappa` | 7.5347 | Concentration of the von Mises distribution around the cardinal axes (κ) |
| `w` | 0.50 | Blend weight of the smeared frame |
| `min_arcmin`, `max_arcmin` | 5, 60 | Range of microsaccade amplitudes, in arcminutes |

Amplitudes are drawn from a lognormal distribution with σ = 0.53 and a median of 19.4 arcminutes, truncated to `[min_arcmin, max_arcmin]`. For other image resolutions `r`, the paper uses `expected_ms = 0.33 + log2(224 / r)`.

## Reproducibility

The transform samples from Python's `random` module and NumPy's global generator, so seeding both, along with PyTorch, to reproduce runs is beneficial.

```python
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

When using multiple data loader workers, seed NumPy in each worker through `worker_init_fn`.

## Citation

```bibtex
@inproceedings{inan2026aughostmentation,
  title     = {AuGhostmentation: The Eyes Never Stand Still---Why Should Neural Networks?},
  author    = {Inan, Emirhan and Arslan, Suayb S.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {To appear}
}
```


## License

This project is released under the MIT License. See `LICENSE` for details.

