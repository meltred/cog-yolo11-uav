> [!NOTE]
> This is [COG](https://cog.run) + [Gradio](gradio) wrapper for [Droneforge's YOLOv11 UAV Finetune model](https://github.com/droneforge/yolov11n-UAV-finetune), intended to be deployed on [Replicate.com](https://replicate.com) and [Hugging Face Spaces](https://huggingface.co/spaces)

Deployed on **Replicate**: https://replicate.com/meltred/yolov11n-uav-finetune

Deployed on **Hugging**: https://huggingface.co/spaces/kunalsin9h/yolov11-uav-finetune

---

# YOLOV11N-UAV FINETUNE

**OPEN SOURCE. SMALL MODEL. UAV DETECTION.**

```
┌─────────────────────────────────────┐
│       TRAINING PERFORMANCE          │
├─────────────────────────────────────┤
│  dj-air3     │ 96.8% mAP50          │
│  uav         │ 80.1% mAP50          │
└─────────────────────────────────────┘
```

█████████████████████████████████████████

## █ CORE

**Lightweight YOLO11n finetuned for unmanned aerial vehicles.**  
**Built for edge deployment.**

## █ SPECS

```
┌─────────────────────────────────────┐
│  MODEL       │ YOLO11n backbone    │
│  INPUT       │ 640x640 RGB         │
│  OUTPUT      │ UAV bounding boxes  │
│  TARGET      │ Mobile/Edge deploy  │
└─────────────────────────────────────┘
```

█████████████████████████████████████████

## █ DATASET

```
████ TRAIN: 6877 images (70%)
████ VALID: 1966 images (20%)
████ TEST:   985 images (10%)
```

**PREPROCESSING**: Auto-orient → Resize to 640x640  
**AUGMENTATIONS**: None applied.

█████████████████████████████████████████

## █ WEIGHTS

```
├── yolov11n-UAV-finetune.pt   [PyTorch]
└── yolov11n-UAV-finetune.onnx [ONNX]
```

█████████████████████████████████████████

## █ CITATION

```bibtex
@misc{
uav-detection-blxxz_dataset,
title = { uav-detection Dataset },
type = { Open Source Dataset },
author = { UAVdetection },
howpublished = { \url{ https://universe.roboflow.com/uavdetection-msr99/uav-detection-blxxz } },
url = { https://universe.roboflow.com/uavdetection-msr99/uav-detection-blxxz },
journal = { Roboflow Universe },
publisher = { Roboflow },
year = { 2024 },
month = { apr },
note = { visited on 2025-08-16 },
}
```

█████████████████████████████████████████

```
███████████████████████████████████████████████████
█ OPEN SOURCE UAV DETECTION FOR ALL              █
███████████████████████████████████████████████████
```

=======---
title: Yolov11 Uav Finetune
emoji: 🏆
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: mit
short_description: Inference for 𖥂 droneforge/yolov11n-UAV-finetune

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
