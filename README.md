# FCG-DETR

## Foreground Confidence-Guided Detection Transformer for Remote Sensing Oriented Object Detection

This is the official implementation of FCG-DETR, a detection transformer architecture designed for oriented object detection in remote sensing images.

## Introduction


We propose FCG-DETR, an efficient oriented object detection
framework exploiting hierarchical foreground guidance, dynamic
token selection and confidence-gated decoding.Specifically, the
Hierarchical Foreground Generation (HFG) module leverages
anisotropic Gaussian modeling to generate category-agnostic
spatial priors, which serve as soft-supervision signals to guide
the encoder toward high-potential regions across varying scales
and aspect ratios.To further optimize efficiency, we design a
Dynamic Foreground Token Selection (DFS) strategy. Instead of
uniform sampling, DFS utilizes normalized cross-scale metrics
to adaptively align token distribution with the target density
of each feature level.Finally, a Foreground Confidence-Gated
Attention (FCGA) mechanism is proposed to resolve the optimization conflict in the decoder.



## Installation

This project is built on [MMRotate 0.x](https://github.com/open-mmlab/mmrotate). Please follow these steps to set up the environment:

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FCG-DETR.git
cd FCG-DETR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install MMRotate:
```bash
pip install -v -e .
```

For more details, please refer to [MMRotate installation guide](https://mmrotate.readthedocs.io/en/latest/install.html).

## Data Preparation

Please refer to [MMRotate data preparation](https://github.com/open-mmlab/mmrotate/blob/main/tools/data/README.md) to prepare the DOTA dataset.

The expected data structure:
```
FCG-DETR
├── data
│   ├── DOTA
│   │   ├── train
│   │   ├── val
│   │   ├── test
```


## Model Zoo

### DOTA 1.0

| Model | Backbone | MS | Sched. | Param. | Input | GFLOPs | FPS | mAP | Download |
|:-----:|:--------:|:--:|:------:|:------:|:-----:|:------:|:---:|:---:|:--------:|
| FCG-DETR | ResNet-50 | - | 2x | - | 1024×1024 | - | - | - | [model](https://pan.baidu.com/s/17bpKv3DSx-6maNoX5a-Fbg)/ [cfg](configs/fcg_detr/dn_fcg_rdetr_r50_dota_summary_train.py) |

**Note**: Download the pretrained model from Baidu Netdisk with password: `6789`

## Training

To train FCG-DETR on DOTA dataset:

```bash
python tools/train.py configs/fcg_detr/dn_fcg_rdetr_r50_dota_summary_train.py
```

## Testing

To evaluate the trained model:

```bash
python tools/test.py configs/fcg_detr/dn_fcg_rdetr_r50_dota_summary_test.py \
    work_dirs/dn_fcg_rdetr_r50_dota/latest.pth \
    --eval mAP
```


## Project Structure

```
FCG-DETR/
├── configs/                  # Configuration files
│   └── fcg_detr/            # FCG-DETR specific configs
├── mmrotate/                # Core implementation
│   ├── models/
│   │   ├── detectors/       # FCG-DETR detector
│   │   └── dense_heads/     # Detection heads
│   ├── datasets/            # Dataset implementations
│   └── core/                # Core utilities
├── tools/                   # Training and testing scripts
│   ├── train.py
│   └── test.py
└── README.md
```

## Main Components

- `mmrotate/models/detectors/fcg_detr.py`: Main FCG-DETR detector implementation
- `mmrotate/models/detectors/fcg_criterion.py`: Foreground confidence criterion with Gaussian modeling
- `mmrotate/models/utils/multi_scale_rotated_deform_atten.py`: SCGA module





## Acknowledgement

This project is based on [MMRotate](https://github.com/open-mmlab/mmrotate). We thank the authors for their excellent work and open-source contribution.

## License

This project is released under the [Apache 2.0 license](LICENSE).


