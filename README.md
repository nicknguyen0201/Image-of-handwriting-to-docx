# Image-of-handwriting-to-docx

This repository supports a final project paper that **compares two handwriting image→text pipelines**:

1. **Traditional OCR pipeline:** CRAFT (text detection) + TrOCR (Transformer OCR recognition)
2. **Vision-language model pipeline:** Qwen2.5-VL (end-to-end transcription via instruction prompting)

[The final paper](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/tree/dev) evaluates both pipelines on:

- **IAM Handwriting** dataset: [TrOCR result printed notebook](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/blob/dev/Final%20Paper%20and%20Collab%20printed%20html/eval_ocr_IAM.pdf), [QWEN result printed notebook](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/blob/dev/Final%20Paper%20and%20Collab%20printed%20html/eval_qwen_IAM.pdf)
- [**Author-collected custom handwriting images**](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/tree/dev/CRAFT-pytorch/tests): the results for the inference can be found in these files for the corresponding images name: [TrOCR](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/blob/dev/Parsed_RealWorld_TrOCR_full/all_text.txt), [Craft+TrOCR](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/blob/dev/Parsed_RealWorld_CRAFT_TrOCR/all_text.txt), [Qwen](https://github.com/nicknguyen0201/Image-of-handwriting-to-docx/blob/dev/Parsed_RealWorld_Qwen/all_text.txt)

The goal is to quantify accuracy and robustness (clean dataset vs. real-world handwriting) and to analyze _why_ a VLM can outperform a detection+recognition OCR stack.

---

## Running the pipelines

### TrOCR-only pipeline (no CRAFT)

Best for **IAM-line** (already line-segmented) and for cases where detection/cropping might fragment text.

Run on a folder:

```bash
python ocr_pipeline.py \
	--input_folder CRAFT-pytorch/tests \
	--use_craft False \
	--output_dir Results_TrOCR_full
```

Run on a single image:

```bash
python ocr_pipeline.py \
	--input_folder path/to/image.png \
	--use_craft False \
	--output_dir Results_TrOCR_single
```

Outputs are written to the `--output_dir` you pass (examples above: `Results_TrOCR_full/`, `Results_TrOCR_single/`):

- `data.csv` (per-image metadata; `word_bboxes` is empty `{}` in this mode)
- `recognized_words.csv` (one row per image; `index=0`, `crop_file=*_full.png`)
- `crops_sorted/` (the saved full-image crop)

### CRAFT + TrOCR pipeline

Best for **cluttered real-world photos** where detecting text regions helps (but can hurt on clean/segmented inputs).

```bash
python ocr_pipeline.py \
	--input_folder CRAFT-pytorch/tests
```

Run on a single image:

```bash
python ocr_pipeline.py \
	--input_folder path/to/image.png \
	--use_craft True \
	--output_dir Results_CRAFT_single
```

Outputs are written to `--output_dir` (default: `./Results/`):

- `data.csv` (per-image detected boxes and metadata)
- `recognized_words.csv` (recognized text for each crop, in reading order)
- `crops_sorted/` (saved crop images used for recognition)

Optional arguments (common):

| Flag             | Default                                   | Description                                     |
| ---------------- | ----------------------------------------- | ----------------------------------------------- |
| `--craft_model`  | `CRAFT-pytorch/weights/craft_mlt_25k.pth` | CRAFT weights path                              |
| `--trocr_model`  | `microsoft/trocr-large-handwritten`       | Hugging Face TrOCR model ID                     |
| `--cuda`         | auto-detected                             | Use CUDA for the CRAFT stage                    |
| `--trocr_device` | `auto`                                    | `auto`, `cpu`, or `cuda` for TrOCR              |
| `--use_craft`    | `True`                                    | `True` = CRAFT+crop, `False` = full-image TrOCR |
| `--output_dir`   | `./Results`                               | Output folder                                   |

### Qwen2.5-VL pipeline

```bash
python qwen_pipeline.py \
	--input_folder CRAFT-pytorch/tests \
	--device cpu
```

Run on a single image:

```bash
python qwen_pipeline.py \
	--input_folder path/to/image.png \
	--device cpu \
	--output_dir Results_Qwen_single
```

Outputs are written to `--output_dir` (default: `./Results_Qwen/`):

- `recognized_text.csv` (transcriptions for every image)
- `<image_stem>.txt` (individual text file per image)

Optional arguments:

| Flag               | Default                       | Description                     |
| ------------------ | ----------------------------- | ------------------------------- |
| `--model_name`     | `Qwen/Qwen2.5-VL-3B-Instruct` | Hugging Face model ID           |
| `--device`         | `auto`                        | `auto`, `cpu`, `cuda`, or `mps` |
| `--max_new_tokens` | `512`                         | Max output length               |
| `--prompt`         | _(built-in prompt)_           | Custom instruction for the VLM  |

---

## Setup

### Install dependencies

Qwen2.5-VL dependencies:

```bash
python -m pip install transformers torch torchvision qwen-vl-utils
```

CRAFT + TrOCR dependencies:

```bash
python -m pip install -r CRAFT-pytorch/requirements.txt
python -m pip install -r CRAFT-pytorch/requirements_trocr.txt
```

---

## Evaluation (IAM)

This repo includes a simple evaluation script for Qwen on IAM lines via Hugging Face datasets:

```bash
python evaluate.py --num_samples 0 --device auto
```

Colab notebook: `eval_qwen.ipynb`

And an equivalent evaluation script for the CRAFT + TrOCR baseline:

```bash
python evaluate_ocr.py --num_samples 0
```

Colab notebook: `eval_ocr.ipynb`

For the paper comparison, mirror the same evaluation procedure for the CRAFT+TrOCR pipeline (same splits, same normalization, same metrics).

---

## Notebooks to use Google Colab GPU (ipynb)

- `eval_qwen.ipynb`: Colab-friendly evaluation of Qwen2.5-VL on IAM lines (same metrics as `evaluate.py`).
- `eval_ocr.ipynb`: Colab-friendly evaluation of TrOCR-only and CRAFT+TrOCR on IAM lines (same metrics as `evaluate_ocr.py`).
