# Project idea: Convert Handwritten notes / manually-filled forms in to docx file

A deep learning pipeline that converts images of handwritten text into digital text using **CRAFT** (text detection) and **TrOCR** (text recognition).

```
Image → Detect Text Regions → Crop Words → Recognize Text → Output
```

---

## Overview

This project builds a two-stage OCR system:

1. **CRAFT** locates where text appears in the image and produces bounding boxes around each word.
2. **TrOCR** reads each cropped word image and outputs the corresponding text.

Both models are Transformer-based, allowing them to handle irregular handwriting, variable spacing, and noisy backgrounds better than traditional CNN-RNN approaches.

---

## Pipeline

### Stage 1 — Text Detection (CRAFT)

CRAFT produces two heatmaps for the input image:

| Map              | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| **Region map**   | Probability each pixel belongs to a character        |
| **Affinity map** | Probability each pixel links two adjacent characters |

**Post-processing steps:**

1. Threshold both maps
2. Combine into a single mask
3. Run connected-component labeling
4. Extract bounding boxes around each word

### Stage 2 — Crop Text Regions

Each bounding box is cropped from the original image.

- **Axis-aligned boxes:** simple array slice `image[y_min:y_max, x_min:x_max]`

### Stage 3 — Text Recognition (TrOCR)

Each cropped word image is passed through TrOCR:

1. **Vision Transformer (ViT) encoder** splits the image into 16×16 patches and encodes them as a sequence of tokens.
2. **Transformer decoder** generates the output text token by token (autoregressive).

```
Cropped image → TrOCRProcessor → pixel_values → model.generate() → decoded text
```

### Stage 4 — Reassembly

Recognized words are sorted top-to-bottom, left-to-right to reconstruct the original document order.

---

## Tech Stack

| Component        | Technology                                |
| ---------------- | ----------------------------------------- |
| Text Detection   | CRAFT                                     |
| Text Recognition | TrOCR (ViT encoder + Transformer decoder) |
| Framework        | PyTorch                                   |
| Image Processing | OpenCV                                    |

---

## Datasets

Possible handwritten images datasets for additional fine-tuning training and evaluation:

- **IAM Handwriting Dataset**
- **VNOnDB** (Vietnamese Handwriting)

---

## Project Structure

```
project/
├── detection/
│   └── craft_model.py       # CRAFT model wrapper & inference
├── recognition/
│   └── trocr_model.py       # TrOCR model wrapper & inference
├── preprocessing/
│   └── crop_utils.py        # Bounding-box cropping & transforms
├── pipeline/
│   └── run_pipeline.py      # End-to-end pipeline script
├── dataset/                  # Data files
└── README.md
```

---

## References

- [CRAFT — Character Region Awareness for Text Detection](https://arxiv.org/abs/1904.01941)
- [TrOCR — Transformer-based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
- [Medium article on how to set up Craft and get coordinate for input of older text recognition model that is CNN + BiLSTM + CTC/Attention](https://medium.com/data-science/pytorch-scene-text-detection-and-recognition-by-craft-and-a-four-stage-network-ec814d39db05)
