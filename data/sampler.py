"""
Scene-aware batch sampler.

The AR-Private-20K dataset contains multiple frames per scene. If frames
from the same scene are randomly spread across training, validation and
test splits, the model can memorize scene-specific features and report
inflated validation performance. To prevent this, all frames from the
same scene are kept in the same split, and the sampler yields batches
that respect scene boundaries.

The sampler groups indices by scene, shuffles scene order (and
within-scene order) at the start of each epoch, then yields indices
scene by scene.
"""
import random
from torch.utils.data import Sampler


class SceneAwareSampler(Sampler):
    """
    Sampler that yields indices grouped by scene.

    Args:
        scene_ids:  list of scene identifiers, one per dataset sample.
        batch_size: target batch size (used for within-scene grouping).
        shuffle:    whether to shuffle the scene order and within-scene
                    order each epoch.
    """

    def __init__(self, scene_ids, batch_size, shuffle=True):
        self.scene_ids = scene_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scene_to_indices = {}
        for i, sid in enumerate(scene_ids):
            self.scene_to_indices.setdefault(sid, []).append(i)
        self.scenes = list(self.scene_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.scenes)

        batches = []
        for sid in self.scenes:
            idxs = self.scene_to_indices[sid][:]
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i:i + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield from b

    def __len__(self):
        return sum(len(v) for v in self.scene_to_indices.values())