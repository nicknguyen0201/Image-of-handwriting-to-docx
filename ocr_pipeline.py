import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image


def get_image_files(folder: str) -> list[Path]:
    """Get all image files from a folder (non-recursive)."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pgm", ".gif"}
    folder_path = Path(folder)
    files: list[Path] = []
    for ext in extensions:
        files.extend(folder_path.glob(f"*{ext}"))
        files.extend(folder_path.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def _add_craft_to_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    craft_dir = repo_root / "CRAFT-pytorch"
    sys.path.insert(0, str(craft_dir))
    return craft_dir


@dataclass
class DetBox:
    score: float
    pts: np.ndarray  # (4,2) float32

    @property
    def x_min(self) -> float:
        return float(self.pts[:, 0].min())

    @property
    def y_min(self) -> float:
        return float(self.pts[:, 1].min())

    @property
    def x_max(self) -> float:
        return float(self.pts[:, 0].max())

    @property
    def y_max(self) -> float:
        return float(self.pts[:, 1].max())

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def h(self) -> float:
        return max(1.0, self.y_max - self.y_min)


def sort_reading_order(boxes: list[DetBox]) -> list[DetBox]:
    if not boxes:
        return []

    heights = np.array([b.h for b in boxes], dtype=np.float32)
    median_h = float(np.median(heights)) if heights.size else 10.0
    line_tol = max(10.0, 0.6 * median_h)

    boxes_sorted_by_y = sorted(boxes, key=lambda b: (b.cy, b.x_min))
    lines: list[dict] = []  # each: {'cy': float, 'items': [DetBox,...]}

    for b in boxes_sorted_by_y:
        best_idx = None
        best_dist = None
        for li, line in enumerate(lines):
            dist = abs(b.cy - line["cy"])
            if dist <= line_tol and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = li

        if best_idx is None:
            lines.append({"cy": b.cy, "items": [b]})
        else:
            line = lines[best_idx]
            line["items"].append(b)
            line["cy"] = float(np.mean([it.cy for it in line["items"]]))

    lines.sort(key=lambda l: l["cy"])
    reading_order: list[DetBox] = []
    for line in lines:
        line["items"].sort(key=lambda b: b.x_min)
        reading_order.extend(line["items"])
    return reading_order


def crop_polygon_with_white_bg(pts_int: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Crop a detected polygon region and place it on a white background.

    This is intentionally defensive: CRAFT can output boxes that slightly fall
    outside image bounds; we clamp the crop and avoid OpenCV bitwise ops that
    can crash on empty arrays.
    """

    if pts_int.size == 0:
        return np.ones((1, 1, 3), dtype=np.uint8) * 255

    h_img, w_img = image_bgr.shape[:2]
    x, y, w, h = cv2.boundingRect(pts_int)

    # Clamp to image bounds
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)

    if x1 <= x0 or y1 <= y0:
        return np.ones((1, 1, 3), dtype=np.uint8) * 255

    cropped = image_bgr[y0:y1, x0:x1].copy()
    if cropped.size == 0:
        return np.ones((1, 1, 3), dtype=np.uint8) * 255

    pts_local = pts_int - np.array([x0, y0], dtype=np.int32)

    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [pts_local], -1, 255, -1, cv2.LINE_AA)

    bg = np.full_like(cropped, 255, dtype=np.uint8)
    m = mask.astype(bool)
    bg[m] = cropped[m]
    return bg


def main() -> None:
    craft_dir = _add_craft_to_path()

    parser = argparse.ArgumentParser(description="CRAFT + TrOCR one-command inference pipeline")

    parser.add_argument(
        "--input_folder",
        default=str(craft_dir / "tests"),
        help="Folder containing input images (or a single image file path)",
    )

    parser.add_argument(
        "--use_craft",
        default=True,
        type=str2bool,
        help="If True, run CRAFT detection + crop before TrOCR. If False, run TrOCR on the full image (no cropping).",
    )
    parser.add_argument(
        "--craft_model",
        default=str(craft_dir / "weights" / "craft_mlt_25k.pth"),
        help="Path to CRAFT weights (.pth)",
    )
    parser.add_argument("--text_threshold", default=0.7, type=float)
    parser.add_argument("--low_text", default=0.4, type=float)
    parser.add_argument("--link_threshold", default=0.4, type=float)
    parser.add_argument("--canvas_size", default=1280, type=int)
    parser.add_argument("--mag_ratio", default=1.5, type=float)
    parser.add_argument("--poly", default=False, action="store_true")
    parser.add_argument("--show_time", default=False, action="store_true")
    parser.add_argument("--cuda", default=str(torch.cuda.is_available()), type=str2bool)

    parser.add_argument(
        "--trocr_model",
        default="microsoft/trocr-large-handwritten",
        help="Hugging Face TrOCR model name",
    )
    parser.add_argument("--trocr_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--trocr_batch_size", type=int, default=8)
    parser.add_argument("--trocr_max_new_tokens", type=int, default=32)

    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "Results"),
        help="Output folder (default: ./Results)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops_sorted"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Load TrOCR
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: transformers. Install with: "
            "python -m pip install -r CRAFT-pytorch/requirements_trocr.txt"
        ) from e

    processor = TrOCRProcessor.from_pretrained(args.trocr_model)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)

    trocr_device = args.trocr_device
    if trocr_device == "auto":
        trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
    if trocr_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --trocr_device cuda but CUDA is not available")

    trocr_model.to(torch.device(trocr_device))
    trocr_model.eval()

    # Resolve input images
    input_path = Path(args.input_folder)
    if input_path.is_file():
        image_list: list[str] = [str(input_path)]
    else:
        # Use our own image glob (works even when not using CRAFT)
        image_list = [str(p) for p in get_image_files(str(input_path))]
    if not image_list:
        raise SystemExit(f"No images found in: {args.input_folder}")

    # Load CRAFT only if needed
    if args.use_craft:
        from craft import CRAFT  # noqa: E402
        import file_utils  # noqa: E402
        import imgproc  # noqa: E402
        import test  # noqa: E402

        net = CRAFT()
        print(f"Loading CRAFT weights: {args.craft_model}")
        if args.cuda:
            if not torch.cuda.is_available():
                raise SystemExit("Requested --cuda True but CUDA is not available")
            net.load_state_dict(test.copyStateDict(torch.load(args.craft_model)))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        else:
            net.load_state_dict(test.copyStateDict(torch.load(args.craft_model, map_location="cpu")))

        net.eval()

    data_csv_path = output_dir / "data.csv"
    recognized_csv_path = output_dir / "recognized_words.csv"

    t_start = time.time()

    # Write both CSVs
    with open(data_csv_path, "w", newline="", encoding="utf-8") as f_data, open(
        recognized_csv_path, "w", newline="", encoding="utf-8"
    ) as f_rec:
        data_writer = csv.DictWriter(f_data, fieldnames=["image_name", "word_bboxes"])
        data_writer.writeheader()

        rec_writer = csv.DictWriter(
            f_rec,
            fieldnames=["image_stem", "index", "score", "crop_file", "text"],
        )
        rec_writer.writeheader()

        for k, image_path in enumerate(image_list):
            print(f"Image {k+1}/{len(image_list)}: {image_path}")
            filename = Path(image_path).stem

            if args.use_craft:
                # CRAFT detection + crop
                image = imgproc.loadImage(image_path)

                bboxes, polys, score_text, det_scores = test.test_net(
                    net,
                    image,
                    args.text_threshold,
                    args.link_threshold,
                    args.low_text,
                    args.cuda,
                    args.poly,
                    args,
                    None,
                )

                # Persist CRAFT outputs
                cv2.imwrite(str(output_dir / f"res_{filename}_mask.jpg"), score_text)
                # saveResult expects a trailing separator
                file_utils.saveResult(
                    image_path, image[:, :, ::-1], polys, dirname=str(output_dir) + os.sep
                )

                det_boxes: list[DetBox] = []
                bbox_score_dict: dict[str, list[list[float]]] = {}
                for score, pts in zip(det_scores, bboxes, strict=False):
                    pts_arr = np.array(pts, dtype=np.float32)
                    if pts_arr.shape != (4, 2):
                        continue
                    det_boxes.append(DetBox(score=float(score), pts=pts_arr))
                    bbox_score_dict[str(float(score))] = pts_arr.tolist()

                data_writer.writerow(
                    {
                        "image_name": Path(image_path).name,
                        "word_bboxes": json.dumps(bbox_score_dict),
                    }
                )

                ordered = sort_reading_order(det_boxes)

                crop_items = []
                crop_images = []
                for idx, b in enumerate(ordered):
                    pts_int = b.pts.astype(np.int32)
                    crop_bgr = crop_polygon_with_white_bg(pts_int, image[:, :, ::-1])
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)

                    crop_file = f"{filename}_{idx:04d}_s{b.score:.6f}.png"
                    pil_img.save(crops_dir / crop_file)

                    crop_items.append((filename, idx, b.score, crop_file))
                    crop_images.append(pil_img)
            else:
                # No cropping: treat whole image as one crop
                pil_img = Image.open(image_path).convert("RGB")
                crop_file = f"{filename}_full.png"
                pil_img.save(crops_dir / crop_file)

                data_writer.writerow(
                    {
                        "image_name": Path(image_path).name,
                        "word_bboxes": json.dumps({}),
                    }
                )

                crop_items = [(filename, 0, 1.0, crop_file)]
                crop_images = [pil_img]

            # TrOCR recognition in batches (keeps reading order)
            bs = max(1, int(args.trocr_batch_size))
            for start in range(0, len(crop_images), bs):
                batch_imgs = crop_images[start : start + bs]
                batch_items = crop_items[start : start + bs]
                inputs = processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(torch.device(trocr_device))
                with torch.no_grad():
                    generated_ids = trocr_model.generate(
                        pixel_values,
                        max_new_tokens=int(args.trocr_max_new_tokens),
                    )
                texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                for (stem, idx, score, crop_file), text in zip(batch_items, texts, strict=False):
                    rec_writer.writerow(
                        {
                            "image_stem": stem,
                            "index": idx,
                            "score": score,
                            "crop_file": crop_file,
                            "text": (text or "").strip(),
                        }
                    )

    print(f"Wrote: {data_csv_path}")
    print(f"Wrote: {recognized_csv_path}")
    print(f"Crops: {crops_dir}")
    print(f"Elapsed: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
