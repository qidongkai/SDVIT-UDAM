"""
SegFormer-B5 teacher wrapper.

The thesis (Table 3.5, Section 3.5.2) specifies the teacher as
SegFormer-B5 trained on Cityscapes, reaching 83.5% mIoU on the
Cityscapes validation set. Loading an ADE20K checkpoint with a randomly
initialised 16-class head would invalidate the teacher and, by extension,
all UDAM experiments, so the teacher must be loaded from a Cityscapes
checkpoint and its 19-class head remapped to the unified 16-category
space.

The wrapper exposes:
    - forward(x): (logits, hidden_states)
    - stage_features(x): the last block of each of the 4 Mix-Transformer
      stages, used for the CKA functional-stage mapping in Section 3.8.7
    - soft_labels(x): teacher softmax probabilities, used by UDAM as P_s
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Cityscapes 19-class train ids -> unified 16-category space.
# This is used to remap the teacher's output head so that its soft labels
# live in the same label space as the student.
CITYSCAPES_19_TO_UNIFIED = {
    0: 0,    # road
    1: 1,    # sidewalk
    2: 2,    # building
    3: 3,    # wall
    4: 4,    # fence
    5: 5,    # pole
    6: 6,    # traffic light
    7: 7,    # traffic sign
    8: 8,    # vegetation
    9: 9,    # terrain
    10: 10,  # sky
    11: 11,  # person
    12: 12,  # rider
    13: 13,  # car
    14: 255, # truck  -> not in unified space
    15: 14,  # bus
    16: 255, # train  -> not in unified space
    17: 15,  # motorcycle -> Two-wheeler
    18: 15,  # bicycle    -> Two-wheeler
}


class SegFormerTeacher(nn.Module):
    """
    Frozen SegFormer-B5 teacher trained on Cityscapes.

    Args:
        pretrained: HuggingFace model id for the Cityscapes-finetuned
                    SegFormer-B5. The default value points to the
                    official checkpoint released by NVIDIA.
        freeze:     whether to freeze all parameters.
    """

    def __init__(
        self,
        pretrained="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        freeze=True,
    ):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        # Load the Cityscapes-finetuned teacher. The model outputs 19
        # train ids; we keep them and remap to the unified space at the
        # soft-label stage.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
        )

        # Register the 19 -> 16 remap as a buffer so it moves with the
        # module across devices.
        remap = torch.full((19,), 255, dtype=torch.long)
        for k, v in CITYSCAPES_19_TO_UNIFIED.items():
            remap[k] = v
        self.register_buffer("remap_19_to_16", remap)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, x):
        """
        Run the teacher and return the raw logits plus hidden states.

        Args:
            x: [B, 3, H, W] input tensor.

        Returns:
            logits: [B, 19, H, W] teacher logits in the 19-class space.
            hidden_states: tuple of 13 tensors (4 stages x 3 blocks + 1
                           embedding).
        """
        out = self.model(pixel_values=x, output_hidden_states=True)
        logits = F.interpolate(
            out.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits, out.hidden_states

    def stage_features(self, x):
        """
        Return the last block of each Mix-Transformer stage.

        Following Section 3.8.7, the four stage features are taken from
        hidden_states[3], [6], [9] and [12].

        Args:
            x: [B, 3, H, W] input tensor.

        Returns:
            list of 4 tensors, each [B, C_i, H_i, W_i].
        """
        _, hs = self.forward(x)
        return [hs[3], hs[6], hs[9], hs[12]]

    def soft_labels(self, x):
        """
        Return the teacher's softmax probabilities in the unified
        16-category space.

        The teacher's 19-class logits are remapped to 16 classes by
        masking out classes that do not exist in the unified space, then
        renormalising. This is the P_s used by UDAM (Section 3.2.2).

        Args:
            x: [B, 3, H, W] input tensor.

        Returns:
            [B, 16, H, W] soft labels.
        """
        logits, _ = self.forward(x)
        B, C, H, W = logits.shape

        # Build a [16, 19] scatter matrix: for each unified class c, sum
        # the logits of the 19-class ids that map to c.
        scatter = torch.zeros(16, 19, device=logits.device, dtype=logits.dtype)
        for k, v in CITYSCAPES_19_TO_UNIFIED.items():
            if v != 255:
                scatter[v, k] = 1.0

        # [B, 16, H, W] = [16, 19] @ [B, 19, H*W]
        flat = logits.flatten(2)                              # [B, 19, H*W]
        unified = torch.einsum("cu,bch->buh", scatter, flat)  # [B, 16, H*W]
        unified = unified.reshape(B, 16, H, W)

        # Renormalise over the 16 unified classes.
        return unified.softmax(dim=1)