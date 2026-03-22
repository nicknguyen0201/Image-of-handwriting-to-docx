"""
Evaluate Qwen2.5-VL on IAM Handwriting Dataset

Loads the Teklia/IAM-line dataset from HuggingFace, runs each image through
the Qwen2.5-VL pipeline, and computes CER and WER against ground truth.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from jiwer import cer, wer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on IAM handwriting")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--output_dir", default="./Evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="Transcribe the handwritten text in this image exactly. Output only the text, nothing else.",
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

    # Load model
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading model: {args.model_name}")
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(args.model_name)

    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )
        model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t_load:.1f}s")

    # Load IAM dataset (all splits combined — no fine-tuning, so no train/test needed)
    from datasets import load_dataset, concatenate_datasets

    print("Loading Teklia/IAM-line dataset (all splits) ...")
    ds_dict = load_dataset("Teklia/IAM-line")
    dataset = concatenate_datasets(list(ds_dict.values()))
    print(f"Total samples in dataset: {len(dataset)}")

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"Evaluating on {len(dataset)} samples")

    # Run inference
    predictions = []
    references = []
    csv_path = output_dir / "eval_results.csv"

    t_start = time.time()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "ground_truth", "prediction", "sample_wer", "sample_cer"])
        writer.writeheader()

        for idx, sample in enumerate(dataset):
            image = sample["image"].convert("RGB")
            ground_truth = sample["text"].strip()

            # Skip empty ground truth
            if not ground_truth:
                continue

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            prediction = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # Per-sample metrics
            sample_wer = wer(ground_truth, prediction) if ground_truth else 0.0
            sample_cer = cer(ground_truth, prediction) if ground_truth else 0.0

            predictions.append(prediction)
            references.append(ground_truth)

            writer.writerow({
                "index": idx,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "sample_wer": f"{sample_wer:.4f}",
                "sample_cer": f"{sample_cer:.4f}",
            })

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(dataset)}] GT: {ground_truth[:50]}  |  PRED: {prediction[:50]}")

    elapsed = time.time() - t_start

    # Compute overall metrics
    total_wer = wer(references, predictions)
    total_cer = cer(references, predictions)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model          : {args.model_name}")
    print(f"Samples        : {len(references)}")
    print(f"WER            : {total_wer:.4f} ({total_wer*100:.2f}%)")
    print(f"CER            : {total_cer:.4f} ({total_cer*100:.2f}%)")
    print(f"Time           : {elapsed:.1f}s ({elapsed/max(len(references),1):.2f}s/sample)")
    print(f"Results CSV    : {csv_path}")
    print("=" * 50)

    # Save summary
    summary_path = output_dir / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Samples: {len(references)}\n")
        f.write(f"WER: {total_wer:.4f} ({total_wer*100:.2f}%)\n")
        f.write(f"CER: {total_cer:.4f} ({total_cer*100:.2f}%)\n")
        f.write(f"Time: {elapsed:.1f}s\n")
    print(f"Summary saved  : {summary_path}")


if __name__ == "__main__":
    main()
