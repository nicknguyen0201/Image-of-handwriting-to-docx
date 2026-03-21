"""\
Evaluate CRAFT + TrOCR on IAM Handwriting Dataset

Loads the Teklia/IAM-line dataset from Hugging Face, runs each image through
CRAFT text detection + TrOCR recognition, and computes CER and WER against
ground truth.

This is the "traditional OCR" baseline counterpart to evaluate.py (Qwen2.5-VL).
"""

import argparse
import csv
import time
from pathlib import Path

import sys
import importlib.util

import numpy as np
import torch
from jiwer import cer, wer
from PIL import Image


def _collapse_spaces(text: str) -> str:
    return " ".join((text or "").split())


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def _add_craft_to_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    craft_dir = repo_root / "CRAFT-pytorch"
    sys.path.insert(0, str(craft_dir))
    return craft_dir


def _load_craft_test_module(craft_dir: Path):
    """Load CRAFT-pytorch/test.py as a uniquely named module.

    This avoids accidentally importing Python's stdlib `test` package.
    """

    test_path = craft_dir / "test.py"
    spec = importlib.util.spec_from_file_location("craft_test", test_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load CRAFT test module at: {test_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["craft_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRAFT + TrOCR on IAM handwriting")

    # Dataset/eval controls
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--output_dir", default="./Evaluation_CRAFT_TrOCR")

    parser.add_argument(
        "--use_craft",
        default=True,
        type=_str2bool,
        help="If True, run CRAFT detection + crop before TrOCR. If False, run TrOCR on the full image (no cropping).",
    )

    # CRAFT
    parser.add_argument(
        "--craft_model",
        default=str(Path(__file__).resolve().parent / "CRAFT-pytorch" / "weights" / "craft_mlt_25k.pth"),
        help="Path to CRAFT weights (.pth)",
    )
    parser.add_argument("--text_threshold", default=0.7, type=float)
    parser.add_argument("--low_text", default=0.4, type=float)
    parser.add_argument("--link_threshold", default=0.4, type=float)
    parser.add_argument("--canvas_size", default=1280, type=int)
    parser.add_argument("--mag_ratio", default=1.5, type=float)
    parser.add_argument("--poly", default=False, action="store_true")
    parser.add_argument("--show_time", default=False, action="store_true")
    parser.add_argument("--cuda", default=torch.cuda.is_available(), type=_str2bool)

    # TrOCR
    parser.add_argument(
        "--trocr_model",
        default="microsoft/trocr-large-handwritten",
        help="Hugging Face TrOCR model name",
    )
    parser.add_argument("--trocr_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--trocr_batch_size", type=int, default=8)
    parser.add_argument("--trocr_max_new_tokens", type=int, default=32)

    args = parser.parse_args()

    use_cuda: bool = bool(args.cuda)
    use_craft: bool = bool(args.use_craft)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TrOCR
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: transformers. Install with: "
            "python -m pip install -r CRAFT-pytorch/requirements_trocr.txt"
        ) from e

    trocr_device = args.trocr_device
    if trocr_device == "auto":
        trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
    if trocr_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --trocr_device cuda but CUDA is not available")

    processor = TrOCRProcessor.from_pretrained(args.trocr_model)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)
    trocr_model.to(torch.device(trocr_device))
    trocr_model.eval()

    # Load CRAFT (optional)
    if use_craft:
        # Import helpers from our existing pipeline (keeps behavior consistent)
        from ocr_pipeline import DetBox, crop_polygon_with_white_bg, sort_reading_order

        craft_dir = _add_craft_to_path()
        from craft import CRAFT  # noqa: E402

        craft_test = _load_craft_test_module(craft_dir)

        net = CRAFT()
        if use_cuda:
            if not torch.cuda.is_available():
                raise SystemExit("Requested --cuda True but CUDA is not available")
            net.load_state_dict(craft_test.copyStateDict(torch.load(args.craft_model)))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            torch.backends.cudnn.benchmark = False
        else:
            net.load_state_dict(
                craft_test.copyStateDict(torch.load(args.craft_model, map_location="cpu"))
            )
        net.eval()

    # Load IAM dataset
    from datasets import load_dataset, concatenate_datasets

    print("Loading Teklia/IAM-line dataset (all splits) ...")
    ds_dict = load_dataset("Teklia/IAM-line")
    dataset = concatenate_datasets(list(ds_dict.values()))
    print(f"Total samples in dataset: {len(dataset)}")

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"Evaluating on {len(dataset)} samples")

    predictions: list[str] = []
    references: list[str] = []

    csv_path = output_dir / "eval_results.csv"
    t_start = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "ground_truth", "prediction", "sample_wer", "sample_cer"],
        )
        writer.writeheader()

        for idx, sample in enumerate(dataset):
            # IAM-line provides PIL.Image in sample["image"]
            pil_img = sample["image"].convert("RGB")
            image_rgb = np.array(pil_img)
            ground_truth = (sample.get("text") or "").strip()

            if not ground_truth:
                continue

            if use_craft:
                # Run CRAFT detection on full image
                bboxes, _polys, _score_text, det_scores = craft_test.test_net(
                    net,
                    image_rgb,
                    args.text_threshold,
                    args.link_threshold,
                    args.low_text,
                    use_cuda,
                    args.poly,
                    args,
                    None,
                )

                # Build DetBox objects and sort into reading order
                det_boxes: list[DetBox] = []
                for score, pts in zip(det_scores, bboxes, strict=False):
                    pts_arr = np.array(pts, dtype=np.float32)
                    if pts_arr.shape != (4, 2):
                        continue
                    det_boxes.append(DetBox(score=float(score), pts=pts_arr))

                ordered = sort_reading_order(det_boxes)

                # Crop each detected region and recognize with TrOCR
                image_bgr = image_rgb[:, :, ::-1]
                crop_images = []
                for b in ordered:
                    pts_int = b.pts.astype(np.int32)
                    crop_bgr = crop_polygon_with_white_bg(pts_int, image_bgr)
                    crop_rgb = crop_bgr[:, :, ::-1]
                    crop_images.append(Image.fromarray(crop_rgb))
            else:
                # No cropping: run TrOCR on the full line image
                crop_images = [pil_img]

            recognized_tokens: list[str] = []
            bs = max(1, int(args.trocr_batch_size))
            for start in range(0, len(crop_images), bs):
                batch_imgs = crop_images[start : start + bs]
                if not batch_imgs:
                    continue
                inputs = processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(torch.device(trocr_device))
                with torch.no_grad():
                    generated_ids = trocr_model.generate(
                        pixel_values,
                        max_new_tokens=int(args.trocr_max_new_tokens),
                    )
                texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                recognized_tokens.extend([(t or "").strip() for t in texts])

            prediction = _collapse_spaces(" ".join([t for t in recognized_tokens if t]))

            sample_wer = wer(ground_truth, prediction) if ground_truth else 0.0
            sample_cer = cer(ground_truth, prediction) if ground_truth else 0.0

            predictions.append(prediction)
            references.append(ground_truth)

            writer.writerow(
                {
                    "index": idx,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "sample_wer": f"{sample_wer:.4f}",
                    "sample_cer": f"{sample_cer:.4f}",
                }
            )

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(dataset)}] GT: {ground_truth[:50]}  |  PRED: {prediction[:50]}")

    elapsed = time.time() - t_start

    total_wer = wer(references, predictions) if references else 0.0
    total_cer = cer(references, predictions) if references else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    if use_craft:
        print("CRAFT enabled   : True")
        print(f"CRAFT model     : {args.craft_model}")
    else:
        print("CRAFT enabled   : False (TrOCR full-image)")
    print(f"TrOCR model     : {args.trocr_model}")
    print(f"Samples         : {len(references)}")
    print(f"WER             : {total_wer:.4f} ({total_wer*100:.2f}%)")
    print(f"CER             : {total_cer:.4f} ({total_cer*100:.2f}%)")
    print(f"Time            : {elapsed:.1f}s ({elapsed/max(len(references),1):.2f}s/sample)")
    print(f"Results CSV     : {csv_path}")
    print("=" * 50)

    summary_path = output_dir / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CRAFT enabled: {use_craft}\n")
        if use_craft:
            f.write(f"CRAFT model: {args.craft_model}\n")
        f.write(f"TrOCR model: {args.trocr_model}\n")
        f.write(f"Samples: {len(references)}\n")
        f.write(f"WER: {total_wer:.4f} ({total_wer*100:.2f}%)\n")
        f.write(f"CER: {total_cer:.4f} ({total_cer*100:.2f}%)\n")
        f.write(f"Time: {elapsed:.1f}s\n")
    print(f"Summary saved   : {summary_path}")


if __name__ == "__main__":
    main()
