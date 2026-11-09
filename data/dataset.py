"""
Dataset for AR semantic segmentation.

Supports Cityscapes, PASCAL VOC 2012 and the private AR-Private-20K
dataset. All datasets are remapped to a unified 16-category label space
following the rules in Table 3.3 of the thesis (Section 3.3.5).

Index file format (one entry per line, space separated):

    <image_path> <label_path> [<scene_id>]

If `scene_id` is omitted, the directory of the image is used as a
fallback scene identifier. The dataset can be instantiated in an
unlabeled mode for UDAM adaptation, in which case a dummy all-255 label
map is returned.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# Table 3.3: Cityscapes -> unified 16-category label space.
# The unified space is:
#   0 Road, 1 Sidewalk, 2 Building, 3 Wall, 4 Fence, 5 Pole,
#   6 Traffic light, 7 Traffic sign, 8 Vegetation, 9 Terrain,
#   10 Sky, 11 Person, 12 Rider, 13 Car, 14 Bus, 15 Two-wheeler
# Two-wheeler merges Cityscapes motorcycle (32) and bicycle (33).
CITYSCAPES_TO_UNIFIED = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    28: 14,  # bus
    32: 15,  # motorcycle -> Two-wheeler
    33: 15,  # bicycle    -> Two-wheeler
}


# Table 3.3: PASCAL VOC -> unified 16-category label space.
# VOC has 21 classes; only the categories that overlap with the unified
# space are mapped, the rest are sent to 255 (ignore).
VOC_TO_UNIFIED = {
    0: 255,   # background
    1: 255,   # aeroplane
    2: 15,    # bicycle -> Two-wheeler
    3: 255,   # bird
    4: 255,   # boat
    5: 255,   # bottle
    6: 14,    # bus
    7: 13,    # car
    8: 255,   # cat
    9: 255,   # chair
    10: 255,  # cow
    11: 255,  # diningtable
    12: 255,  # dog
    13: 255,  # horse
    14: 15,   # motorbike -> Two-wheeler
    15: 11,   # person
    16: 255,  # pottedplant
    17: 255,  # sheep
    18: 255,  # sofa
    19: 255,  # train
    20: 255,  # tvmonitor
}

# Table 3.3: AR-Private-20K -> unified label space.
# The private dataset is already annotated in the unified space, so the
# remap is the identity for categories present in the dataset. Bus is
# target-missing and never appears.
AR_TO_UNIFIED = {i: i for i in range(16)}


class ARSegDataset(Dataset):
    """
    AR semantic segmentation dataset with unified label space.

    Args:
        root:       root directory containing index files
                    (train.txt / val.txt / test.txt).
        split:      one of {"train", "val", "test"}.
        transform:  callable applied to the PIL image.
        unlabeled:  if True, return a dummy label map and skip the label
                    image, used for source-free UDAM adaptation.
        domain:     one of {"cityscapes", "voc", "ar"}. Selects the
                    correct label remap table.
    """

    def __init__(self, root, split="train", transform=None,
                 unlabeled=False, domain="cityscapes"):
        self.root = root
        self.split = split
        self.transform = transform
        self.unlabeled = unlabeled
        self.domain = domain
        self.samples, self.scene_ids = self._load_index()

        if domain == "cityscapes":
            self._mapping = CITYSCAPES_TO_UNIFIED
        elif domain == "voc":
            self._mapping = VOC_TO_UNIFIED
        elif domain == "ar":
            self._mapping = AR_TO_UNIFIED
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _load_index(self):
        """
        Read the split index file and return the list of samples along
        with their scene identifiers.
        """
        samples, scene_ids = [], []
        list_path = os.path.join(self.root, f"{self.split}.txt")
        with open(list_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img, lbl = parts[0], parts[1]
                scene = parts[2] if len(parts) > 2 else os.path.dirname(img)
                samples.append((img, lbl))
                scene_ids.append(scene)
        return samples, scene_ids

    def _remap(self, lbl):
        """
        Remap dataset-specific ids to the unified 16-category space.
        Categories not present in the mapping are sent to 255 (ignore).
        """
        arr = np.array(lbl)
        out = np.full_like(arr, 255)
        for k, v in self._mapping.items():
            out[arr == k] = v
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, lbl_path = self.samples[i]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        x = self.transform(img) if self.transform else img

        if self.unlabeled:
            H = x.shape[-2] if torch.is_tensor(x) else 512
            return x, torch.full((H, H), 255, dtype=torch.long)

        lbl = self._remap(Image.open(os.path.join(self.root, lbl_path)))
        return x, torch.from_numpy(lbl).long()