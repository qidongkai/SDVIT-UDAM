# SDVIT-UDAM

# SDViT-UDAM: Efficient Knowledge Distillation for AR Semantic Segmentation

This repository provides the reference implementation of the **Static-Dynamic collaborative Vision Transformer (SDViT)**, the **Unsupervised Domain Adaptation distillation Module (UDAM)**, and the **Distillation Efficiency Ratio (DER)** metric described in the thesis *Image Semantic Segmentation in Augmented Reality: Knowledge Distillation Technology*.

The framework targets real-time, privacy-preserving semantic segmentation on AR end-side hardware. It is designed around three coupled constraints that are not simultaneously satisfied by existing methods:

- **Latency**: end-to-end segmentation inference under the 20 ms AR interaction budget.
- **Precision**: pixel-level edge and small-object fidelity required for virtual-real fusion.
- **Data privacy**: no raw source-domain data and no target-domain pixel annotations during adaptation.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Repository Structure](#repository-structure)
3. [Requirements](#requirements)
4. [Datasets](#datasets)
5. [Configuration](#configuration)
6. [Training SDViT](#training-sdvit)
7. [UDAM Source-Free Adaptation](#udam-source-free-adaptation)
8. [Evaluation](#evaluation)
9. [Ablation Study](#ablation-study)
10. [Edge Deployment](#edge-deployment)
11. [Reproducibility Protocol](#reproducibility-protocol)
12. [Citations](#citations)
13. [FAQ](#faq)

---

## Highlights

- **SDViT**: A static-dynamic collaborative ViT backbone. The deepest encoder path is never pruned and serves as the self-distillation target. Intermediate layers are pruned layer by layer with an adaptive threshold derived from scene complexity, then aligned back to the deep features through a projection function.
- **UDAM**: A source-free unsupervised domain adaptation module. Only frozen teacher parameters and unlabeled target images are used during adaptation. The source-domain knowledge is stored as a compact, non-invertible feature bank (800 vectors, ~2.5 MB).
- **DER**: A composite metric that unifies parameter compression (PCR), computational reduction (CCR) and performance retention (PRR) through a weighted geometric mean.
- **Statistical rigor**: All experiments are repeated across ten pre-declared seeds, with paired tests and Holm-Bonferroni correction within each comparison family.
- **Deployment pipeline**: ONNX export, operator compatibility check, SNPE (Snapdragon) and TensorRT (AR glasses) engine builders, latency and power measurement, and statistical validation of deployment metrics.

---

## Repository Structure

```
sdvit_udam/
├── configs/
│   ├── __init__.py
│   ├── default.yaml
│   └── loader.py
├── models/
│   ├── __init__.py
│   ├── vit_backbone.py
│   ├── dynamic_pruner.py
│   ├── projector.py
│   ├── sdvit.py
│   ├── teacher.py
│   └── seg_head.py
├── losses/
│   ├── __init__.py
│   ├── sd_loss.py
│   ├── udam_loss.py
│   └── reliability.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── sampler.py
│   └── scene_check.py
├── eval/
│   ├── __init__.py
│   ├── metrics.py
│   ├── cka.py
│   ├── ood.py
│   ├── der_extreme.py
│   └── stats.py
├── deploy/
│   ├── __init__.py
│   ├── export_onnx.py
│   ├── check_ops.py
│   ├── build_snpe.py
│   ├── build_tensorrt.py
│   ├── measure_latency.py
│   ├── measure_power.py
│   └── stats_validate.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── logger.py
│   └── vis.py
├── train_sdvit.py
├── train_udam.py
├── eval_ablation.py
├── run_all.sh
├── requirements.txt
└── README.md
```

---

## Requirements

The training stack follows the configuration in Section 3.5.1 of the thesis:

| Component | Version |
|---|---|
| Python | 3.9.16 |
| PyTorch | 1.13.1 |
| TorchVision | 0.14.1 |
| CUDA | 11.7 |
| cuDNN | 8.5.0 |
| timm | >= 0.9.0 |
| transformers | >= 4.30 |
| onnx / onnxruntime | >= 1.14 / >= 1.15 |

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For deployment, the following external tools must be available on the build machine:

- **Qualcomm SNPE SDK** (version compatible with Snapdragon 8 Gen 2).
- **NVIDIA TensorRT** (>= 8.5, with `trtexec`).

The paths to these SDKs are configured in `configs/default.yaml` under `deploy.snapdragon_snpe` and `deploy.ar_glasses_trt`.

---

## Datasets

The framework is evaluated on three datasets:

### Cityscapes

- Download from [https://www.cityscapes-dataset.com](https://www.cityscapes-dataset.com).
- Split: 2,975 training / 500 validation / 1,525 test images.
- Used as the source domain for teacher pre-training and as a benchmark for horizontal comparisons.

### PASCAL VOC 2012

- Download from [http://host.robots.ox.ac.uk/pascal/VOC](http://host.robots.ox.ac.uk/pascal/VOC).
- Split: 1,464 training / 1,449 validation / 1,456 test images.
- Used as an auxiliary source domain for cross-domain experiments.

### AR-Private-20K

This dataset is not publicly released. It contains 20,000 first-person AR scene images collected from wearable and vehicle-mounted AR terminals, covering urban road navigation, indoor wearable interaction, and industrial inspection.

To use the training pipeline, prepare the dataset in the following layout:

```
AR-Private-20K/
├── images/
│   ├── scene_0001_frame_0001.jpg
│   └── ...
├── labels/
│   ├── scene_0001_frame_0001.png
│   └── ...
├── train.txt
├── val.txt
├── test.txt
└── scene_meta.json
```

Each index file (`train.txt`, `val.txt`, `test.txt`) contains one entry per line:

```
<image_path> <label_path> <scene_id>
```

For UDAM adaptation, the training split is loaded in unlabeled mode: the label path is still read but the label tensor is replaced with all-255.

### Unified Label Space

All datasets are remapped to the 16-category unified label space defined in Table 3.3 of the thesis:

```
0 Road, 1 Sidewalk, 2 Building, 3 Wall, 4 Fence, 5 Pole,
6 Traffic light, 7 Traffic sign, 8 Vegetation, 9 Terrain,
10 Sky, 11 Person, 12 Rider, 13 Car, 14 Bus, 15 Two-wheeler
```

The remapping tables for Cityscapes and PASCAL VOC are defined in `data/dataset.py`.

---

## Configuration

All hyperparameters are defined in `configs/default.yaml`. Key fields:

### Model

- `embed_dim`: 384
- `depth`: 6
- `num_heads`: 6
- `patch_size`: 16
- `img_size`: 512
- `prune_layers`: [1, 2, 3, 4]
- `alpha_attn`: 1.0 (fixed balance hyperparameter in Eq 3.2)
- `lambda_sd`: 0.3 (self-distillation weight, grid-search optimum)
- `min_keep_ratio`, `max_keep_ratio`: 0.35, 0.85

### Training

- `batch_size_per_gpu`: 8
- `grad_accum_steps`: 2 (effective batch size 16)
- `epochs`: 120
- `lr`: 1e-4, `min_lr`: 1e-6
- `weight_decay`: 1e-5
- `amp`: true
- `clip_grad`: 1.0
- `seeds`: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

### UDAM

- `tau_p`: 0.9, `tau_H`: 1.5
- `rho_min`: 0.1, `fallback_floor`: 0.7
- `alpha_mmd`: 1.0, `gamma_floor`: 0.2
- `K_views`: 4
- `per_class_bank`: 50, `feature_bank_size`: 800

### Deployment

- `onnx_opset`: 17
- `snapdragon_snpe`, `ar_glasses_trt`: paths to external SDKs

---

## Training SDViT

The supervised training of SDViT is implemented in `train_sdvit.py`.

### Single-seed training

```bash
python train_sdvit.py --cfg configs/default.yaml --seed 0
```

### What happens during training

1. The student model is initialized from scratch (random weights).
2. A scene-aware sampler groups the training data by scene identifier, preventing scene-level data leakage.
3. At each layer in `prune_layers`, the token scorer computes importance scores and physically prunes the token sequence to a variable length.
4. The unpruned deep feature `f_deep` is used as the target for the self-distillation loss, with a token reliability mask computed from cosine similarity.
5. The total loss is `L = L_seg + 0.3 * L_sd`.
6. The model is evaluated on the training split each epoch, and the best checkpoint is saved to `runs/sdvit_seed{seed}.pth`.

### Output

- `runs/sdvit_seed{seed}.pth`: best checkpoint.
- `runs/sdvit_seed{seed}.log`: training log.

### Ten-seed training

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python train_sdvit.py --cfg configs/default.yaml --seed ${seed}
done
```

The ten-seed protocol is part of the reproducibility commitment in Section 4.1.4 of the thesis.

---

## UDAM Source-Free Adaptation

The UDAM module is implemented in `train_udam.py`.

### Prerequisites

- A trained SDViT checkpoint from the supervised stage (`runs/sdvit_seed{seed}.pth`).
- The frozen teacher model (SegFormer-B5, loaded from a Cityscapes-finetuned HuggingFace checkpoint).
- The AR-Private-20K training split, used in unlabeled mode.

### Run a single seed

```bash
python train_udam.py --cfg configs/default.yaml --seed 0
```

### What happens during adaptation

1. A source-domain feature bank is built once from the frozen teacher. For each of the 16 unified categories, up to 50 teacher feature vectors are collected, capped at 800. The bank is stored as a fixed tensor and is not invertible to an image.
2. The target loader returns K=4 views per image (original + 3 augmented).
3. For each target batch:
   - The teacher produces soft labels on the target image (this is `p_s`, the source-domain prior transferred to the target scene).
   - The student produces multi-view features and predictions.
   - The four-stage reliability filter (pixel-level, sample-level, augmentation consistency, temporal stability) determines which target pixels contribute to the loss.
   - MMD between the source bank and target features determines `gamma`.
   - The loss combines the KL divergence and multi-view consistency, weighted by `gamma` and scaled by the retained pixel ratio `rho`.
4. The adapted student is saved to `runs/sdvit_udam_seed{seed}.pth`.

### Ten-seed adaptation

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python train_udam.py --cfg configs/default.yaml --seed ${seed}
done
```

### Important note

UDAM is source-free in the following sense: no raw source-domain image is loaded during the adaptation loop, and no source image can be reconstructed from anything retained. The only source-domain artifact is the 800-vector feature bank, which is non-invertible. This definition is stated explicitly in Section 4.3.0 of the thesis.

---

## Evaluation

### Segmentation metrics

The `SegMetrics` class in `eval/metrics.py` computes mIoU, pixel accuracy and per-class IoU. It handles the ignore index (255) and is used throughout training and evaluation.

### DER composite index

The `der` function implements the weighted geometric mean from Eq 3.26:

```python
from eval.metrics import der
score = der(pcr_value, ccr_value, prr_value, w1=0.3, w2=0.5, w3=0.2)
```

The weights `(0.3, 0.5, 0.2)` reflect the AR deployment priority: latency (CCR) is the veto constraint, storage (PCR) is auxiliary, and accuracy retention (PRR) is guaranteed by the SDViT compensation mechanism.

### CKA representation alignment

The `compute_cka_all` function in `eval/cka.py` computes layer-wise CKA between the student and the teacher, using the functional-stage mapping defined in Section 3.8.7:

| Student layer | Teacher stage |
|---|---|
| 1, 2 | Stage 1 |
| 3 | Stage 2 |
| 4 | Stage 3 |
| 5, 6 | Stage 4 |

The mapping is approximate and based on feature resolution and semantic abstraction, not on strict layer-to-layer correspondence.

### OOD evaluation

The `eval/ood.py` module provides:

- `expected_calibration_error`: Expected Calibration Error with 15 equal-width bins.
- `group_by_condition`: Grouping of samples by illumination, weather or occlusion condition.

These are used in the OOD analysis of Section 4.5.2.

---

## Ablation Study

The ablation study is implemented in `eval_ablation.py`. Six groups from Table 4-14 are supported:

| Variant | Pruning | Self-distillation | Threshold |
|---|---|---|---|
| `full` | adaptive | yes | adaptive |
| `no_prune` | no | no | adaptive |
| `prune_only` | adaptive | no | adaptive |
| `sd_only` | no | yes | adaptive |
| `lambda_zero` | adaptive | yes (`lambda_sd=0`) | adaptive |
| `fixed_thr` | fixed (0.58) | yes | fixed |

Run a single variant:

```bash
python eval_ablation.py --cfg configs/default.yaml --variant full --seed 0
```

Run all variants:

```bash
for variant in full no_prune prune_only sd_only lambda_zero fixed_thr; do
  python eval_ablation.py --cfg configs/default.yaml --variant ${variant} --seed 0
done
```

---

## Edge Deployment

The deployment pipeline follows the six steps in Section 3.5.5 of the thesis.

### Step 1: ONNX export

```bash
python -m deploy.export_onnx
```

The exported graph uses opset 17 (native LayerNormalization support) and has a dynamic batch axis.

### Step 2: Operator compatibility check

```bash
python -m deploy.check_ops
```

Verifies that MatMul, Softmax, LayerNormalization, Conv and GELU are present in the exported graph. If GELU is decomposed into `Erf`-based primitive operations, this is expected for TensorRT >= 11.0, where the deprecated GELU plugin was removed.

### Step 3: Engine build

For Snapdragon (INT8 via SNPE):

```bash
python -m deploy.build_snpe
```

For AR glasses (FP16 via TensorRT):

```bash
python -m deploy.build_tensorrt
```

LayerNorm layers are forced to FP32 because FP16 LayerNorm introduces a larger accuracy loss in a preliminary check.

### Step 4: Latency measurement

```bash
python -m deploy.measure_latency
```

Runs 10 warm-up iterations followed by 100 timed iterations, repeated across 10 independent runs, and reports the mean, standard deviation, and 95% confidence interval of the per-run medians.

### Step 5: Power measurement

```bash
python -m deploy.measure_power
```

Measures average power consumption during 60 seconds of continuous inference at 30 FPS using the device's on-board power monitoring interface.

### Step 6: Statistical validation

```bash
python -m deploy.stats_validate
```

Reports the mean, standard deviation and 95% confidence interval for any given list of measurements. Pairwise comparisons use the same test selection rule as Section 4.1.4: paired t-test when normality holds, Wilcoxon signed-rank test otherwise, with Holm-Bonferroni correction within each comparison family.

---

## Reproducibility Protocol

The reproducibility commitments of the thesis are implemented as follows:

### Ten-seed protocol

Every experimental configuration is repeated across the ten pre-declared seeds `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`. The seeds control dataset shuffling, model initialization, augmentation sampling and batch ordering. No seed is added or removed after the experiments begin.

### Statistical testing

Pairwise comparisons use the test selection rule in `utils/stats.py`:

1. If the Shapiro-Wilk test on the paired differences does not reject normality (`p > 0.05`), use the paired t-test.
2. Otherwise, use the Wilcoxon signed-rank test.

The Holm-Bonferroni correction is applied within each table's comparison family, not across the entire chapter.

### Scene-aware splitting

The AR-Private-20K dataset uses a scene-aware split: all frames from the same scene are assigned to the same split. The `check_scene_split` function verifies that:

- No scene identifier appears in more than one split.
- The per-split categorical distribution deviates from the overall distribution by no more than 5 percentage points for each metadata dimension.

### Environment locking

The training stack versions are pinned in `requirements.txt`. The Docker image digest is recorded separately in the version-control repository, because package versions alone do not determine numerical behavior — the underlying BLAS libraries and cuDNN algorithm selection can vary across builds.

### End-to-end script

`run_all.sh` runs the full pipeline: ten seeds of SDViT training, ten seeds of UDAM adaptation, six ablation variants, and the deployment pipeline.

```bash
bash run_all.sh
```

---

## Citations

If you use this code in your research, please cite:

```bibtex
@phdthesis{qidongkai2026sdvit,
  title  = {Image Semantic Segmentation in Augmented Reality: Knowledge Distillation Technology},
  author = {Qi, Dongkai},
  school = {Asia Pacific University of Technology \& Innovation},
  year   = {2026}
}
```

Related works referenced in the thesis:

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. arXiv:1503.02531.
- Wang, D., et al. (2021). *TENT: Fully Test-Time Adaptation by Entropy Minimization*. ICLR.
- Liang, J., Hu, D., & Feng, J. (2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*. ICML.
- Tang, Y., et al. (2026). *DynamicViT* (reference for token pruning).
- Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS.

---

## FAQ

### Q1: Why is AR-Private-20K not included?

The dataset contains first-person AR scene images collected in privacy-sensitive environments (indoor office spaces, personal wearable first-person scenes, private vehicle driving scenes). It is subject to a data usage agreement and cannot be publicly redistributed. The dataset construction, annotation protocol, and ethical approval are documented in Section 3.3.3 and Section 3.3.4 of the thesis.

### Q2: Why does the teacher use a Cityscapes-finetuned checkpoint rather than a generic one?

The teacher must produce meaningful source-domain knowledge. Loading a generic ADE20K checkpoint with a randomly initialized 16-class head would invalidate the teacher and, by extension, all UDAM experiments. The correct procedure is to load the Cityscapes-finetuned SegFormer-B5 and remap its 19-class output to the unified 16-category space, as implemented in `models/teacher.py`.

### Q3: What does "source-free" mean in this repository?

"Source-free" means that no raw source-domain image is loaded during the adaptation loop, and no source image can be reconstructed from anything retained. The only source-domain artifact is the 800-vector feature bank stored in `train_udam.py`, which is not invertible to an image. This definition is stated explicitly in Section 4.3.0 of the thesis.

### Q4: Why is the scene complexity proxy not exactly the foreground fraction?

The thesis defines scene complexity as the fraction of pixels belonging to foreground categories (anything other than sky, road, building, vegetation, and terrain). At inference time, no ground-truth labels are available, so the proxy is computed from the semantic energy of the patch embeddings, which correlates with the foreground fraction. This is a deployment-time approximation; during training, the true foreground fraction can be computed from the labels and used as a reference. The approximation is documented in `models/sdvit.py`.

### Q5: Why are some layers kept in FP32 during TensorRT deployment?

FP16 LayerNorm introduces a larger accuracy loss than FP16 convolution or attention. In the TensorRT build, LayerNorm layers are forced to FP32 using `--layerPrecisions=*LayerNorm*:fp32`, following Section 3.5.5 of the thesis.

### Q6: Why does the AR glasses platform not meet the 20 ms budget at 512×512?

The AR glasses SoC is less capable than the Snapdragon 8 Gen 2. At 512×512 with FP16 and LayerNorm in FP32, the mean latency is 21.3 ms, which exceeds the 20 ms budget. The field trial therefore uses a reduced resolution of 384×384, at which the mean latency is 17.9 ms with an mIoU cost of 1.6 points relative to 512×512. This limitation is reported explicitly in Section 4.5.4 of the thesis, not adjusted away.

### Q7: Can I use this code with a different teacher?

Yes. The teacher is loaded through the `SegFormerTeacher` class in `models/teacher.py`. To use a different teacher, you need to:

1. Replace the `SegformerForSemanticSegmentation` loader with the new teacher's loader.
2. Update the remapping table from the teacher's label space to the 16-category unified space.
3. Update the functional-stage mapping in `eval/cka.py` if the new teacher has a different number of stages.

### Q8: How do I run a single variant of the ablation study without training?

The ablation script assumes a trained checkpoint exists at `runs/sdvit_seed{seed}.pth`. If you have already trained the student, you can run any ablation variant directly:

```bash
python eval_ablation.py --cfg configs/default.yaml --variant prune_only --seed 0
```

The script does not retrain the model; it reconfigures the forward pass and evaluates the existing checkpoint under the new configuration.

---

## License

This code is released for academic research use only. Commercial use, redistribution of the AR-Private-20K dataset, and deployment in production AR systems require explicit permission from the author and the data provider.

---

## Contact

For questions about the paper, the code, or the dataset, please contact the author:

**Qi Dongkai**  
Asia Pacific University of Technology & Innovation (APU)  
Graduate School of Technology

---

*Last updated: 2026*
