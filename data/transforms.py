"""
Image transforms used by SDViT and UDAM.

- build_train_transform: deterministic training transform for the
  segmentation branch.
- MultiViewTransform: returns K views (1 original + K-1 augmented) for
  the multi-view self-supervised consistency loss in UDAM.
"""
import torch
import torchvision.transforms as T


def build_train_transform(size=512):
    """
    Standard training transform for supervised segmentation.

    Applies resizing, random horizontal flipping, mild color jitter, and
    ImageNet normalization.

    Args:
        size: target image resolution.

    Returns:
        A torchvision.Compose transform.
    """
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class MultiViewTransform:
    """
    Multi-view transform that produces K views of the same image.

    The first view is the un-augmented (resized and normalized) image.
    The remaining K-1 views apply randomized color / perspective / blur
    augmentation before resize and normalization. These views feed the
    augmentation-consistency filter and the multi-view self-supervised
    loss in UDAM.

    Args:
        size: target image resolution.
        K:    total number of views (must be >= 1).
    """

    def __init__(self, size=512, K=4):
        assert K >= 1, "K must be >= 1"
        self.K = K
        self.base = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.aug = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.RandomApply([T.GaussianBlur(3)], p=0.3),
        ])

    def __call__(self, img):
        views = [self.base(img)]
        for _ in range(self.K - 1):
            views.append(self.base(self.aug(img)))
        return torch.stack(views)