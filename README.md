# Industrial Anomaly Detection

This project builds on [EfficientAD](https://github.com/nelson1425/EfficientAD), an unofficial implementation of:

- [EfficientAD: Accurate and Efficient Anomaly Detection for Industrial Applications](https://arxiv.org/abs/2303.14535)
- [CDO: Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization](https://arxiv.org/abs/2302.08769)

---

## Overview

Anomaly detection identifies defects in industrial products without requiring labeled defect examples during training. This project extends **EfficientAD** with several enhancements: Collaborative Discrepancy Optimization (CDO) to replace the standard ImageNet OOD penalty, synthetic anomaly injection during training, appearance augmentation, image tiling for high-resolution inputs, and a configurable α hyperparameter for inference-time blending of the student and autoencoder anomaly maps.

The system is validated primarily on the **Real-IAD plastic-nut** sub-dataset and generalizes to other Real-IAD and MVTec AD categories with minimal fine-tuning.

### Key features

- Teacher-student architecture with a pretrained, frozen PDN (Patch Description Network) teacher distilled from WideResNet-101
- Convolutional autoencoder branch (64-dim bottleneck) for reconstruction-based anomaly scoring
- Collaborative Discrepancy Optimization (CDO) via synthetic anomaly injection and a dual-margin loss that widens the gap between normal and anomalous feature responses
- Configurable α to blend student and autoencoder anomaly maps at inference time
- Appearance augmentation (ColorJitter, sharpness, gamma) and synthetic noise-patch injection (40 % probability, 1–4 patches, 10–50 px)
- Image tiling strategy for high-resolution inputs: tiles are processed independently and fused into a coherent global anomaly map
- Optional ImageNet out-of-distribution penalty during training
- Intermediate evaluation checkpoints every 10 000 iterations
- Timestamped logging to file and console
- MVTec AD evaluation scripts computing AU-PRO and AU-ROC

---

## Repository structure

```
AnomalyDetection/
├── efficientad.py                  # Full model: teacher + student + autoencoder
├── efficientad_no_autoencoder.py   # Ablation: teacher + student only
├── common.py                       # PDN and autoencoder model definitions
├── pretraining.py                  # Pretrain the teacher PDN on ImageNet
├── benchmark.py                    # Inference speed benchmark
├── logger_setup.py                 # Timestamped logger (file + console)
├── run_efficientad.sh              # Example training launch script
├── test_efficientad.sh             # Example evaluation launch script
├── models/
│   ├── teacher_small.pth           # Pretrained small PDN teacher weights
│   └── teacher_medium.pth          # Pretrained medium PDN teacher weights
├── results/
│   ├── mvtec_ad_small.json         # Stored results for small model
│   └── mvtec_ad_medium.json        # Stored results for medium model
├── mvtec_ad_evaluation/            # Official MVTec AD evaluation scripts
│   ├── evaluate_experiment.py      # Single-experiment AU-PRO / AU-ROC
│   ├── evaluate_multiple_experiments.py
│   ├── print_metrics.py
│   └── ...
└── prediction.ipynb                # Notebook for visualizing predictions
```

---

## Architecture

The system uses three components. The teacher is always frozen; only the student and autoencoder are trained.

| Component | Role |
|-----------|------|
| **Teacher (PDN)** | Frozen pretrained network. Provides the reference feature representation of normal images. Distilled from WideResNet-101 on ImageNet. |
| **Student (PDN, 2× channels)** | Trained to mimic the teacher on normal images. High teacher-student discrepancy indicates an anomaly. The second half of the student channels is used for the STAE consistency loss. |
| **Autoencoder** | Convolutional encoder-decoder with a 64-dimensional latent bottleneck. Trained to reconstruct teacher features from augmented inputs. High reconstruction error indicates an anomaly. |

### PDN model sizes

| Architecture | Parameters | AU-PRO (plastic-nut) |
|-------------|-----------|----------------------|
| Small PDN   | 1,096,320  | 84.79 %              |
| Medium PDN  | 11,615,488 | 92.03 %              |

The small PDN has a 33×33 receptive field and outputs 384 channels. It is well-suited for hardware-constrained or real-time deployments. The medium PDN offers a better trade-off between detection accuracy and real-time feasibility.

### Anomaly map fusion

Both branches produce a per-pixel anomaly score. The maps are quantile-normalized (lower quantile q_a → 0, upper quantile q_b → 0.1) and fused using a blending weight α:

```
map_combined = α · map_st + (1 − α) · map_ae
```

α defaults to `0.5`. It can be adjusted at inference time to shift emphasis between classification accuracy (student map) and localization precision (autoencoder map).

The no-autoencoder variant (`efficientad_no_autoencoder.py`) uses only `map_st`.

### Loss terms

| Flag | Loss | Description |
|------|------|-------------|
| `--coeff_hard` | `loss_hard` | Mean squared error on the top-0.1% hardest teacher-student distances |
| `--coeff_penalty` | `loss_penalty` | Out-of-distribution penalty (ImageNet images or CDO synthetic anomalies) |
| `--coeff_ae` | `loss_ae` | Teacher vs. autoencoder reconstruction error |
| `--coeff_stae` | `loss_stae` | Autoencoder output vs. student second-half channels |

With CDO enabled the total loss is:

```
L_total = L_Hard + L_CDO + L_AE + L_STAE
```

All coefficients default to `1.0`.

---

## Datasets

| Dataset | Description |
|---------|-------------|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 15 industrial object categories, ~5 354 images total, ~250–600 training images per category |
| [MVTec LOCO](https://www.mvtec.com/company/research/datasets/mvtec-loco) | 5 categories with logical anomalies |
| [Real-IAD](https://realiad4ad.github.io/Real-IAD/) | 30 objects, 150 000+ multi-view images (5 camera angles); primary test case: **plastic-nut** (2 511 normal, 1 246 abnormal images; defect types: missing parts, contamination, scratch, pit) |

Optional: [ImageNet](https://www.image-net.org/) training set for the OOD pretraining penalty.

---

## Setup

### Requirements

- Python 3.10+
- PyTorch (CUDA optional but recommended; tested on NVIDIA RTX 4000 Ada)
- torchvision, numpy, scikit-learn, tifffile, tqdm

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn tifffile tqdm
```

For the evaluation scripts (Python 3.7+):

```bash
pip install -r mvtec_ad_evaluation/requirements.txt
# or with conda:
conda env create --name mad_eval --file=mvtec_ad_evaluation/conda_environment.yml
conda activate mad_eval
```

---

## Training

### Full model (teacher + student + autoencoder)

```bash
python efficientad.py \
    --dataset mvtec_ad \
    --subdataset bottle \
    --output_dir output/bottle_run1 \
    --model_size small \
    --weights models/teacher_small.pth \
    --mvtec_ad_path /path/to/mvtec_anomaly_detection \
    --train_steps 70000
```

To enable the ImageNet OOD penalty:

```bash
python efficientad.py \
    --imagenet_train_path /path/to/ImageNet/train \
    ...
```

### Ablation — no autoencoder

```bash
python efficientad_no_autoencoder.py \
    --dataset mvtec_ad \
    --subdataset bottle \
    --output_dir output/bottle_noae \
    --model_size small \
    --weights models/teacher_small.pth \
    --mvtec_ad_path /path/to/mvtec_anomaly_detection \
    --train_steps 70000
```

### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-d / --dataset` | `mvtec_ad` | `mvtec_ad` or `mvtec_loco` |
| `-s / --subdataset` | `bottle` | Sub-dataset name |
| `-o / --output_dir` | `output/1` | Directory for saved models and anomaly maps |
| `-m / --model_size` | `small` | `small` or `medium` |
| `-w / --weights` | `models/teacher_small.pth` | Path to pretrained teacher weights |
| `-i / --imagenet_train_path` | `none` | ImageNet train path (`none` disables penalty) |
| `-a / --mvtec_ad_path` | `./mvtec_anomaly_detection` | MVTec AD dataset root |
| `-b / --mvtec_loco_path` | `./mvtec_loco_anomaly_detection` | MVTec LOCO dataset root |
| `-t / --train_steps` | `70000` | Total number of training iterations |
| `-l / --log_path` | `./logs` | Directory for log files |
| `--coeff_hard` | `1.0` | Weight for hard teacher-student loss |
| `--coeff_penalty` | `1.0` | Weight for ImageNet OOD penalty loss |
| `--coeff_ae` | `1.0` | Weight for autoencoder reconstruction loss |
| `--coeff_stae` | `1.0` | Weight for student-autoencoder consistency loss |

### Training details

- Optimizer: Adam, lr = 1 × 10⁻⁴, weight decay = 1 × 10⁻⁵
- LR scheduler: StepLR with decay applied at 95 % of total training steps
- Batch size: 1 (main training loop)
- Input resolution: 256 × 256
- Normalization: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

---

## Data Augmentation

Three augmentation strategies are applied during training:

| Strategy | Details |
|----------|---------|
| **Default** | Resize to 256 × 256, ImageNet normalization |
| **Appearance** | Random ColorJitter (brightness, contrast, saturation, hue), random sharpness, random gamma correction |
| **Synthetic anomalies** | Applied with 40 % probability; injects 1–4 rectangular noise patches of 10–50 px into the image to simulate surface defects |

---

## Outputs

After training, the following are written under `--output_dir`:

```
output/
└── <run_name>/
    ├── trainings/<dataset>/<subdataset>/
    │   ├── teacher_tmp.pth       # Checkpoint (every 1000 steps)
    │   ├── student_tmp.pth
    │   ├── autoencoder_tmp.pth
    │   ├── teacher_final.pth
    │   ├── student_final.pth
    │   └── autoencoder_final.pth
    └── anomaly_maps/<dataset>/<subdataset>/test/
        └── <defect_class>/
            └── <image_id>.tiff   # Per-pixel anomaly score maps
```

Intermediate AUC scores are logged every 10 000 iterations. Final AUC (image-level ROC) is printed at the end.

---

## Evaluation

### Single experiment

```bash
python mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir /path/to/mvtec_anomaly_detection \
    --anomaly_maps_dir ./output/<run_name>/anomaly_maps/mvtec_ad/ \
    --output_dir ./output/<run_name>/metrics/mvtec_ad/ \
    --evaluated_objects bottle
```

Metrics computed:
- **AU-PRO** — area under the per-region overlap curve (localization quality)
- **AU-ROC** — area under the ROC curve (image-level classification)
- **Optimal threshold** and **best accuracy**
- **F-score** at the optimal threshold

### Multiple experiments

```bash
python mvtec_ad_evaluation/evaluate_multiple_experiments.py \
    --dataset_base_dir /path/to/mvtec_anomaly_detection \
    --experiment_configs mvtec_ad_evaluation/experiment_configs.json \
    --output_dir ./output/metrics/
```

### Print results table

```bash
python mvtec_ad_evaluation/print_metrics.py --metrics_folder ./output/<run_name>/metrics/
```

---

## Performance Results

### Ablation study — Real-IAD plastic-nut (medium PDN, 2 259 training images)

| Configuration | AUROC | AU-PRO |
|--------------|-------|--------|
| Autoencoder only | 91.66 % | 94.80 % |
| Student only | 97.60 % | 73.68 % |
| Student + Autoencoder | **97.85 %** | **92.03 %** |
| Student + Autoencoder + ImageNet OOD | 96.18 % | 80.86 % |

Distillation (teacher-student) and reconstruction (autoencoder) are complementary: the student branch delivers strong image-level classification while the autoencoder improves pixel-level localization. The ImageNet OOD penalty degrades performance on plastic-nut; CDO-based synthetic anomalies are preferred.

### Effect of training data size — Real-IAD plastic-nut

| Training images | AUROC | AU-PRO |
|----------------|-------|--------|
| 2 259 | 97.85 % | 92.03 % |
| 500   | 96.07 % | 95.48 % |
| 100   | 92.82 % | 95.67 % |

The model is robust under data scarcity: 500 training images yield competitive detection and even better localization than the full set, suggesting the autoencoder generalizes well from limited normal examples.

### Cross-dataset generalization

| Dataset | AUROC | AU-PRO |
|---------|-------|--------|
| Plastic-nut (Real-IAD) | 97.07 % | 95.48 % |
| USB (Real-IAD) | 97.16 % | 94.11 % |
| Metal-nut (MVTec AD) | 99.26 % | 94.14 % |
| Cable (MVTec AD) | 96.92 % | 83.93 % |
| Hazelnut (MVTec AD) | 96.82 % | 81.80 % |

Results demonstrate that the architecture and training strategies generalize well to other industrial defect types with minimal fine-tuning.

---

## Teacher pretraining

If you want to pretrain your own teacher PDN on ImageNet (instead of using the provided weights):

```bash
python pretraining.py --output_folder output/pretraining/
```

Edit `pretraining.py` to set `model_size` (`small` or `medium`) and `imagenet_train_path` before running. The script trains for 60 000 iterations using a Wide ResNet-101-2 as the feature extractor backbone. It first computes per-channel mean and variance over 10 000 samples for feature normalization, then trains the PDN to minimize MSE against the normalized backbone features.

---

## Speed benchmark

```bash
python benchmark.py
```

Runs 2000 forward passes with random input (256 × 256) and prints the mean inference time over the last 1000 runs. Uses FP16 on GPU automatically.

---

## Logging

All training runs write a timestamped log file to `--log_path` (default: `./logs/`). Example filename: `log_2024-10-18_12-30-45.txt`. Output is mirrored to the console.

---

## Future directions

- **DINO-based teacher pretraining**: replace WideResNet-101 with a DINO Vision Transformer (ViT) to improve feature generalization without labeled data
- **Adaptive tiling**: dynamically determine tile size and count based on image content or defect scale
- **Advanced tile fusion**: attention-weighted or edge-aware blending to reduce false positives at tile boundaries
- **Multi-scale analysis**: operate the teacher-student model at multiple resolutions simultaneously

---

## License

See [LICENSE](LICENSE) for the main project. The MVTec AD evaluation scripts are covered by a separate license in [mvtec_ad_evaluation/LICENSE.txt](mvtec_ad_evaluation/LICENSE.txt).

---

## Citation

If you use this work, please cite the original papers:

```bibtex
@article{batzner2023efficientad,
  title={EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies},
  author={Batzner, Kilian and Heckler, Lars and König, Rebecca},
  journal={arXiv preprint arXiv:2303.14535},
  year={2023}
}

@article{cao2023collaborative,
  title={Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization},
  author={Cao, Yunkang and Xu, Xiaohao and Liu, Zhaoge and Shen, Weiming},
  journal={IEEE Transactions on Industrial Informatics},
  volume={19},
  number={11},
  pages={10674--10683},
  year={2023}
}

@article{wang2024realiad,
  title={Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection},
  author={Wang, Chengjie and others},
  journal={arXiv preprint arXiv:2403.12580},
  year={2024}
}
```
