# CANSURF: An ASV-View Can Dataset for Detection and Tracking of Surface-Level Debris

This repository contains the raw CANSURF dataset and the augmentation script used to produce the full training set described in the paper. The repository contains the latest version of the dataset. To view all the versions of the dataset, please visit the Zenodo archive of the dataset: https://doi.org/10.5281/zenodo.20057287

**Paper:** Z. Aljundi, A. Moosa, M. Elemam, and Z. F. Rahmatullah, "CANSURF: An ASV-View Can Dataset and Benchmark for Detection and Tracking of Surface-Level Debris," *2025 8th International Conference on Signal Processing and Information Security (ICSPIS)*, 2025. DOI: [10.1109/ICSPIS67605.2025.11318414](https://doi.org/10.1109/ICSPIS67605.2025.11318414)

---

## Repository Structure

```
CANSURF/
├── dataset/
│   ├── train/
│   │   ├── images/       # Raw training images (.jpg)
│   │   └── labels/       # YOLO-format annotation files (.txt)
│   ├── val/
│   │   ├── images/       # Validation images (.jpg)
│   │   └── labels/       # Validation annotation files (.txt)
│   └── data.yaml         # YOLO dataset config
├── augment.py            # Augmentation script
└── README.md
```

---

## Dataset

The raw dataset contains 7,171 images of aluminum cans floating on water. Annotations are in YOLO format (bounding boxes).

For the full augmented dataset used in the paper, run the augmentation script as is below without any augmentation type modification.

The raw dataset files are hosted on Zenodo: [insert DOI link]

---

## Augmentation

The augmentation script (`augment.py`) applies 10 augmentation types to each training image, expanding the training set. The validation set is left unaugmented to preserve evaluation integrity.

### Augmentation Types

| Type | Description |
|------|-------------|
| `bright` | Random brightness increase (factor 1.2–2.0) to simulate high sunlight |
| `dark` | Random brightness reduction (factor 0.2–0.8) to simulate low-light or shadow |
| `color` | Random contrast and color saturation boost to simulate camera variation |
| `noise` | Gaussian noise with random mean and standard deviation to simulate sensor noise |
| `blur` | Gaussian blur with random radius (1–4) to simulate motion or focus issues |
| `weather` | Random fog effect via Albumentations to simulate adverse weather |
| `compress` | JPEG compression at low quality (10–30) to simulate degraded footage |
| `mosaic` | Pixelation via downscale/upscale to simulate low-resolution conditions |
| `hflip` | Horizontal flip with corrected bounding box coordinates |
| `vflip` | Vertical flip with corrected bounding box coordinates |

Flip augmentations correctly adjust YOLO bounding box coordinates. All other augmentations copy the original label file unchanged, as they do not affect object geometry.

### Usage

**Install dependencies:**
```bash
pip install pillow numpy albumentations opencv-python
```

```bash
python augment.py
```

The augmented dataset will be written to `./augmentation_data/`.

### Customising Augmentations

**To change which augmentations are applied**, edit the `augmentation_types` list in the `main()` function:

```python
augmentation_types = [
    'bright', 'dark', 'color', 'noise', 'blur',
    'weather', 'compress', 'mosaic', 'hflip', 'vflip'
]
```

Remove any types you don't want. Each entry in this list is applied once to every training image.

It is recommended to correspondigly remove any unwanted augmentation types in the `augmentation_functions` dictionary in `augment_image_and_label()`:

```python
augmentation_functions = {
    'bright': lambda i, l: adjust_brightness(i, l, bright=True),
    'dark':   lambda i, l: adjust_brightness(i, l, bright=False),
    'color':  adjust_color,
}
```
---

## Citation

If you use CANSURF in your research, please cite:

```bibtex
@INPROCEEDINGS{11318414,
  author={Aljundi, Zaid and Moosa, Abdullah and Elemam, Mostafa and Rahmatullah, Zahra F.},
  booktitle={2025 8th International Conference on Signal Processing and Information Security (ICSPIS)}, 
  title={CANSURF: An ASV-View Can Dataset and Benchmark for Detection and Tracking of Surface-Level Debris}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Weather;Wind;Aluminum;Pipelines;Detectors;Object detection;Benchmark testing;Water pollution;Reliability;Videos;Autonomous Surface Vehicles (ASV);Marine Debris Detection;Object Detection;Benchmark Datasets;Small Object Detection;Multi-Object Tracking;Domain Adaptation;Real-time systems},
  doi={10.1109/ICSPIS67605.2025.11318414}}

```

