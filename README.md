# WSSNet Release

This repository contains source code for:

- AdaAttReNet self-supervised SAR despeckling
- WSSNet
- WSSNet-Mini
- WSSNet-Tiny
- training and inference scripts
- comparison-experiment runners
- unified evaluation code

Third-party datasets are not redistributed in this repository. Obtain datasets from their original sources and organize them as described below.

## Repository Layout

```text
AdaAttReNet/
  config.yaml
  train.py
  evaluate.py
  predict.py
  models/
  datasets/
  losses/
  requirements.txt
WSSNet/
  configs/
  models/
  tools/
  utils/
  requirements.txt
comparison_experiments/
  baseline_sources.yaml
  protocol.yaml
  prepare_baselines.py
  run_comparison.py
  evaluate_predictions.py
```

## AdaAttReNet

AdaAttReNet uses Noise2Noise-style self-supervised reconstruction training for SAR despeckling. The network contains AttUNet, PCAUNet, and MaskNet. AttUNet includes attention modules. MaskNet adaptively fuses the two denoising branches. The training loss includes the main L1 loss for the final fused output and auxiliary L1 losses for AttUNet and PCAUNet, with weights of 1:0.1:0.1.

Model definition:

```text
AdaAttReNet/models/ada_att_re_net.py
AdaAttReNet/models/att_unet.py
AdaAttReNet/models/pca_unet.py
AdaAttReNet/models/mask_net.py
```

Configure `AdaAttReNet/config.yaml` before running. The relevant fields are:

```yaml
data:
  train_dir: path/to/despeckling/train_images
  val_dir: path/to/despeckling/val_images
  test_dir: path/to/despeckling/test_images
  crop_size: 256
training:
  output_dir: outputs
  batch_size: 4
  epochs: 100
  learning_rate: 0.001
```

Install dependencies:

```bash
cd AdaAttReNet
pip install -r requirements.txt
```

Train:

```bash
python train.py --config config.yaml
```

Evaluate a checkpoint on the validation directory configured in `config.yaml`:

```bash
python evaluate.py --config config.yaml --checkpoint outputs/speckle-HHMM/checkpoint-epoch100-VALLOSS.pt
```

Predict:

```bash
python predict.py --config config.yaml --checkpoint outputs/speckle-HHMM/checkpoint-epoch100-VALLOSS.pt --input-dir data/test --output-dir outputs/predictions
```

AdaAttReNet checkpoints are written under `training.output_dir`. If `checkpoint_overwrite` is false, the script creates a subdirectory named from the noise type and current time; otherwise it writes `checkpoint-<noise_type>.pt`.

## WSSNet

This repository contains WSSNet, WSSNet-Mini, and WSSNet-Tiny.

Model definitions:

```text
WSSNet/models/wssnet.py
WSSNet/models/wssnet_mini.py
WSSNet/models/wssnet_tiny.py
```

The implemented WSSNet modules include:

- MSSAtt: Multi-Scale Split Attention, implemented in `WSSNet/models/modules/attention.py`
- DE-RWE: Dynamic Enhanced Residual Wavelet Encoder, implemented as `DERWE` in `WSSNet/models/modules/wavelet.py`
- MEMCAU: Mamba-Enhanced Multi-scale Context Aggregation Upsampler, implemented in `WSSNet/models/modules/decoder.py`

Install dependencies:

```bash
cd WSSNet
pip install -r requirements.txt
```

WSSNet training expects a data root containing paired image and mask directories:

```text
data/
  images/
    sample_001.png
  masks/
    sample_001.png
```

Supported image and mask formats are PNG, JPG/JPEG, BMP, TIF, and TIFF. Masks are binary segmentation masks; pixel values greater than 127 are treated as foreground. Images and masks are paired by file stem.

Configuration files:

```text
WSSNet/configs/wssnet.yaml
WSSNet/configs/wssnet_mini.yaml
WSSNet/configs/wssnet_tiny.yaml
```

The default WSSNet configuration uses RGB input, 256 x 256 input size, batch size 8, 100 epochs, Adam optimizer, initial learning rate 0.0001, seed 11, and two classes.

Train:

```bash
python tools/train.py --model WSSNet --config configs/wssnet.yaml --data-root data --save-dir weights/wssnet
```

For the smaller variants, use the corresponding model name and config:

```bash
python tools/train.py --model WSSNet-Mini --config configs/wssnet_mini.yaml --data-root data --save-dir weights/wssnet_mini
python tools/train.py --model WSSNet-Tiny --config configs/wssnet_tiny.yaml --data-root data --save-dir weights/wssnet_tiny
```

Training writes `last_epoch_weights.pth` and `best_epoch_weights.pth` to `--save-dir`.

Predict:

```bash
python tools/predict.py --model WSSNet --checkpoint weights/wssnet/best_epoch_weights.pth --input-dir data/images --output-dir pred_masks/wssnet --visualization-dir vis_masks/wssnet
```

Evaluate predicted masks:

```bash
python tools/evaluate.py --pred-dir pred_masks/wssnet --gt-dir data/masks --output-dir metric_csv/wssnet --model-name WSSNet
```

The unified evaluation reports OA, Precision, Recall, F1, IoU, Kappa, Specificity, FPR, FNR, and F2.

## Comparison Experiments

| Model        | Year | Original implementation                               |
| ------------ | ---: | ----------------------------------------------------- |
| U-Net        | 2015 | https://github.com/bubbliiiing/unet-pytorch           |
| DeepLabV3+   | 2018 | https://github.com/bubbliiiing/deeplabv3-plus-pytorch |
| TransUNet    | 2021 | https://github.com/Beckschen/TransUNet                |
| EMCAD        | 2024 | https://github.com/SLDGroup/EMCAD                     |
| SelfReg-UNet | 2024 | https://github.com/ChongQingNoSubway/SelfReg-UNet     |
| EfficientViM | 2025 | https://github.com/mlvlab/EfficientViM                |
| TransOilSeg  | 2025 | https://github.com/cy7372/TransOilSeg                 |
| OilSpillNet  | 2026 | https://github.com/AnavKatwal/OilSpillNet             |

Third-party source code is not redistributed in this repository. The scripts in comparison_experiments provide the dataset configuration, unified experimental settings, execution interfaces, and evaluation procedure used in the comparison experiments.

Prepare or locate the external baseline repositories outside this release package:

```bash
python comparison_experiments/prepare_baselines.py --third-party-root ../third_party_baselines
```

Run a comparison model after its external command templates or adapter have been configured:

```bash
python comparison_experiments/run_comparison.py --model unet
python comparison_experiments/run_comparison.py --model deeplabv3plus
python comparison_experiments/run_comparison.py --model transunet
python comparison_experiments/run_comparison.py --model emcad
python comparison_experiments/run_comparison.py --model selfreg_unet
python comparison_experiments/run_comparison.py --model efficientvim
python comparison_experiments/run_comparison.py --model transoilseg
python comparison_experiments/run_comparison.py --model oilspillnet
```

Evaluate predictions with the unified metrics:

```bash
python comparison_experiments/evaluate_predictions.py --pred-dir comparison_outputs/unet/predictions --gt-dir data/test/masks --output-dir comparison_outputs/unet/metrics --model-name unet
```

## Data

The release does not include third-party data. Obtain the required data from its original source and organize it with relative paths before running the scripts.

For WSSNet training and direct prediction:

```text
WSSNet/data/
  images/
    sample_001.png
  masks/
    sample_001.png
```

For comparison experiments, `comparison_experiments/protocol.yaml` uses the following split layout:

```text
data/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

For AdaAttReNet despeckling, set `train_dir`, `val_dir`, and `test_dir` in `AdaAttReNet/config.yaml` to directories containing SAR images. Supported image formats are PNG, JPG/JPEG, BMP, TIF, and TIFF.

## Environment

AdaAttReNet dependencies are listed in `AdaAttReNet/requirements.txt`:

```text
numpy>=1.21
Pillow>=9.0
PyYAML>=6.0
torch>=1.10
torchvision>=0.11
```

WSSNet dependencies are listed in `WSSNet/requirements.txt`:

```text
torch
numpy
opencv-python
Pillow
PyYAML
tqdm
pytorch-wavelets
```