"""
Qwen2.5-VL OCR Pipeline

A simplified OCR pipeline using Qwen2.5-VL-3B vision-language model.
Unlike the CRAFT+TrOCR pipeline, this model handles both text detection
and recognition in a single forward pass.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from PIL import Image


def get_image_files(folder: str) -> list[Path]:
    """Get all image files from a folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    folder_path = Path(folder)
    files = []
    for ext in extensions:
        files.extend(folder_path.glob(f"*{ext}"))
        files.extend(folder_path.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL OCR pipeline for handwritten text recognition"
    )

    parser.add_argument(
        "--input_folder",
        default="./CRAFT-pytorch/tests",
        help="Folder containing input images",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face model name for Qwen2.5-VL",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--output_dir",
        default="./Results_Qwen",
        help="Output folder for results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--prompt",
        default="Read all the handwritten text in this image. Transcribe it exactly as written, preserving the original formatting and line breaks where possible.",
        help="Prompt to send to the model",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Load model and processor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: transformers. Install with:\n"
            "pip install transformers torch torchvision qwen-vl-utils"
        ) from e

    print(f"Loading model: {args.model_name}")
    t_load = time.time()

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Load model with appropriate settings for the device
    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
    elif device == "mps":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )
        model = model.to(device)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )

    model.eval()
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    # Get input images
    image_list = get_image_files(args.input_folder)
    if not image_list:
        raise SystemExit(f"No images found in: {args.input_folder}")

    print(f"Found {len(image_list)} images")

    results_csv_path = output_dir / "recognized_text.csv"
    t_start = time.time()

    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "transcription"])
        writer.writeheader()

        for idx, image_path in enumerate(image_list):
            print(f"Processing {idx + 1}/{len(image_list)}: {image_path.name}")

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Prepare the conversation format for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]

            # Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = inputs.to(device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                )

            # Decode only the generated tokens (exclude input)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            transcription = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Write to CSV
            writer.writerow({
                "image_name": image_path.name,
                "transcription": transcription.strip(),
            })

            # Also save individual text files
            text_file = output_dir / f"{image_path.stem}.txt"
            with open(text_file, "w", encoding="utf-8") as tf:
                tf.write(transcription.strip())

            print(f"  -> {len(transcription.split())} words recognized")

    print(f"\nResults saved to: {results_csv_path}")
    print(f"Individual text files in: {output_dir}")
    print(f"Total time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
