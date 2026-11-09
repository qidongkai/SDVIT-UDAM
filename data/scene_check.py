"""
Scene-aware split validation.

Implements the checks required in Section 4.1.2 of the thesis:
    - no scene id appears in more than one split;
    - for each metadata dimension (e.g. scene type, illumination,
      weather, occlusion), the per-split categorical distribution
      deviates from the overall distribution by less than `max_dev`.
"""
from collections import Counter


def check_scene_split(scene_ids_by_split, meta_by_split, max_dev=0.05):
    """
    Verify that scene ids are disjoint across splits and that the
    per-split categorical distribution deviates from the overall
    distribution by no more than `max_dev`.

    Args:
        scene_ids_by_split: dict mapping split name -> list of scene ids.
        meta_by_split:      dict mapping split name -> dict of metadata
                            dimension -> list of categorical values.
        max_dev:            maximum allowed absolute deviation of the
                            per-split fraction from the overall fraction.

    Raises:
        AssertionError if a scene id appears in more than one split or
        if any categorical deviation exceeds `max_dev`.
    """
    sets = [set(ids) for ids in scene_ids_by_split.values()]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            assert not inter, f"scene leak between splits: {inter}"

    for key in meta_by_split[list(meta_by_split)[0]].keys():
        total = Counter()
        for _, meta in meta_by_split.items():
            total.update(meta[key])
        total_n = sum(total.values())

        for split, meta in meta_by_split.items():
            cnt = Counter(meta[key])
            n = sum(cnt.values())
            for cat, c in cnt.items():
                frac_split = c / n
                frac_total = total[cat] / total_n
                dev = abs(frac_split - frac_total)
                assert dev <= max_dev, (
                    f"{key}/{cat} in split '{split}': deviation={dev:.3f} "
                    f"exceeds max_dev={max_dev}"
                )

    print("Scene split check: PASS")