"""
Donut Fine-Tuning on FUNSD via LoRA (v3)
=========================================
Resumes from v2 merged model with fresh LoRA adapter.
More data (10x augmentation), higher capacity (rank 32), longer training (20 epochs).

Hardware: RTX 3060 Ti (8GB VRAM)
Dataset: FUNSD (149 train + 10x augmentation → ~1,500 samples)

v3 changes from v2:
- Resume from v2 merged model (builds on prior learning)
- 10x augmentation (was 4x) → ~1,500 training samples
- LoRA rank 32 (was 16) with alpha=64
- 20 epochs (was 10) with lower LR 1e-4
- Added all linear layers as LoRA targets
"""

import json
import os
import re
import sys
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# Config
# ============================================================

SEED = 42
MODEL_ID = "./donut-funsd-merged-v2"   # Resume from v2 merged model
BASE_MODEL_ID = "naver-clova-ix/donut-base"  # For fresh start fallback
OUTPUT_DIR = "./donut-funsd-lora-v3"
MERGED_DIR = "./donut-funsd-merged-v3"

MAX_LENGTH = 768          # Covers 97% of targets
IMAGE_SIZE = [1280, 960]  # Donut default
BATCH_SIZE = 1            # 8GB VRAM constraint
GRAD_ACCUM = 8            # Effective batch size = 8
EPOCHS = 20               # Extended training (v2 was 10)
LR = 1e-4                 # Lower LR for fine-tuning on top of v2 (v2 was 2e-4)
LORA_R = 32               # Higher rank for more capacity (v2 was 16)
AUG_FACTOR = 10           # 10x augmentation (v2 was 4x)

TASK_START_TOKEN = "<s_funsd>"
TASK_END_TOKEN = "</s_funsd>"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FUNSD_DIR = Path("data/funsd")


# ============================================================
# Step 1: Parse FUNSD annotations
# ============================================================

def parse_funsd_annotation(annotation_path: str) -> dict:
    """Extract key-value pairs from FUNSD annotation."""
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    elements = {item['id']: item for item in data['form']}

    link_map = defaultdict(set)
    for item in data['form']:
        for link in item.get('linking', []):
            from_id, to_id = link
            from_elem = elements.get(from_id)
            to_elem = elements.get(to_id)
            if from_elem and to_elem:
                if from_elem['label'] == 'question' and to_elem['label'] == 'answer':
                    link_map[from_id].add(to_id)
                elif from_elem['label'] == 'answer' and to_elem['label'] == 'question':
                    link_map[to_id].add(from_id)
                elif from_elem['label'] == 'header' and to_elem['label'] == 'answer':
                    link_map[from_id].add(to_id)

    kv_pairs = {}
    for q_id, a_ids in link_map.items():
        question = elements[q_id]
        q_text = question['text'].strip().rstrip(':').strip()
        if not q_text or q_text == ':':
            continue
        answers = []
        for a_id in a_ids:
            a_text = elements[a_id]['text'].strip()
            if a_text:
                answers.append(a_text)
        if answers:
            kv_pairs[q_text] = '; '.join(answers) if len(answers) > 1 else answers[0]

    return kv_pairs


def load_split(split: str) -> list:
    """Load FUNSD split as list of {image, ground_truth, doc_id}."""
    split_dir = 'training_data' if split == 'training' else 'testing_data'
    ann_dir = FUNSD_DIR / 'dataset' / split_dir / 'annotations'
    img_dir = FUNSD_DIR / 'dataset' / split_dir / 'images'

    dataset = []
    for ann_file in sorted(ann_dir.glob('*.json')):
        doc_id = ann_file.stem
        img_file = img_dir / f'{doc_id}.png'
        if not img_file.exists():
            continue
        kv_pairs = parse_funsd_annotation(str(ann_file))
        if not kv_pairs:
            continue
        # Truncate long values to keep within token limit
        truncated = {}
        for k, v in kv_pairs.items():
            if len(v) > 200:
                v = v[:200]
            truncated[k] = v
        dataset.append({
            "image": str(img_file),
            "ground_truth": json.dumps({"gt_parse": truncated}, ensure_ascii=False),
            "doc_id": doc_id,
        })
    return dataset


# ============================================================
# Step 2: Augmentation
# ============================================================

def augment_dataset(dataset: list, augmentations_per_sample: int = 4) -> list:
    """Augment with Augraphy."""
    import augraphy

    aug_dir = Path("data/augmented_v3")
    aug_dir.mkdir(exist_ok=True)

    # Simple pipeline that won't crash
    def make_pipeline():
        return augraphy.AugraphyPipeline(
            ink_phase=[
                augraphy.Brightness(brightness_range=(0.8, 1.2), p=0.5),
            ],
            paper_phase=[
                augraphy.NoiseTexturize(sigma_range=(2, 5), turbulence_range=(2, 5), p=0.5),
            ],
            post_phase=[
                augraphy.Jpeg(quality_range=(50, 90), p=0.5),
            ],
        )

    augmented = list(dataset)
    total = len(dataset)

    for i, sample in enumerate(dataset):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Augmenting {i+1}/{total}...")

        img = np.array(Image.open(sample['image']).convert('RGB'))

        for j in range(augmentations_per_sample):
            try:
                pipeline = make_pipeline()
                aug_img = pipeline(img)
                aug_path = aug_dir / f"{sample['doc_id']}_aug{j}.png"
                Image.fromarray(aug_img).save(str(aug_path))
                augmented.append({
                    "image": str(aug_path),
                    "ground_truth": sample['ground_truth'],
                    "doc_id": f"{sample['doc_id']}_aug{j}",
                })
            except Exception:
                pass  # skip failed augmentations

    return augmented


# ============================================================
# Step 3: Dataset class
# ============================================================

class FUNSDDonutDataset(Dataset):
    def __init__(self, data: list, processor: DonutProcessor, max_length: int = MAX_LENGTH):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = Image.open(sample['image']).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        gt = json.loads(sample['ground_truth'])
        target_text = (
            TASK_START_TOKEN
            + json.dumps(gt['gt_parse'], ensure_ascii=False)
            + TASK_END_TOKEN
        )

        labels = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  DONUT FINE-TUNING ON FUNSD")
    print("=" * 60)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model: {MODEL_ID}")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"  Max length: {MAX_LENGTH}")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading FUNSD data...")
    train_data = load_split('training')
    test_data = load_split('testing')
    print(f"  Training: {len(train_data)} docs")
    print(f"  Testing:  {len(test_data)} docs")

    # --- Augment ---
    print(f"\n[2/6] Augmenting training data ({AUG_FACTOR}x)...")
    aug_dir = Path("data/augmented_v3")
    expected_count = len(train_data) * AUG_FACTOR
    if aug_dir.exists() and len(list(aug_dir.glob("*.png"))) >= expected_count * 0.9:
        print("  Using cached v3 augmentations...")
        train_augmented = list(train_data)
        for sample in train_data:
            for j in range(AUG_FACTOR):
                aug_path = aug_dir / f"{sample['doc_id']}_aug{j}.png"
                if aug_path.exists():
                    train_augmented.append({
                        "image": str(aug_path),
                        "ground_truth": sample['ground_truth'],
                        "doc_id": f"{sample['doc_id']}_aug{j}",
                    })
    else:
        train_augmented = augment_dataset(train_data, augmentations_per_sample=AUG_FACTOR)
    print(f"  Augmented: {len(train_augmented)} samples ({len(train_augmented)/len(train_data):.1f}x)")

    # --- Load model ---
    print(f"\n[3/6] Loading model from {MODEL_ID}...")
    if Path(MODEL_ID).exists():
        print("  Resuming from v2 merged model...")
        processor = DonutProcessor.from_pretrained(MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    else:
        print("  Starting fresh from donut-base...")
        processor = DonutProcessor.from_pretrained(BASE_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_ID)

    # Ensure task tokens exist
    existing_tokens = processor.tokenizer.get_vocab()
    new_tokens = [t for t in [TASK_START_TOKEN, TASK_END_TOKEN] if t not in existing_tokens]
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_START_TOKEN)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # --- Apply LoRA ---
    print("\n[4/6] Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=64,           # 2x rank for stronger adaptation
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],  # All key layers
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Create datasets ---
    train_dataset = FUNSDDonutDataset(train_augmented, processor)
    test_dataset = FUNSDDonutDataset(test_data, processor)

    # --- Train ---
    print(f"\n[5/6] Training ({EPOCHS} epochs)...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,  # Faster training
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time

    print(f"\n  Training complete!")
    print(f"  Loss: {train_result.training_loss:.4f}")
    print(f"  Time: {train_time/60:.1f} minutes")

    # --- Save ---
    print(f"\n[6/6] Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Merge LoRA into base model
    print("  Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_DIR)
    processor.save_pretrained(MERGED_DIR)

    total_size = sum(
        os.path.getsize(os.path.join(MERGED_DIR, f))
        for f in os.listdir(MERGED_DIR)
        if os.path.isfile(os.path.join(MERGED_DIR, f))
    ) / (1024 * 1024)
    print(f"  Merged model size: {total_size:.0f} MB")
    print(f"  Saved to: {MERGED_DIR}")

    # --- Quick evaluation ---
    print("\n" + "=" * 60)
    print("  QUICK EVALUATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    merged_model.to(device)
    merged_model.eval()

    json_valid = 0
    total_time = 0
    results = []

    for i, sample in enumerate(test_data[:10]):  # Quick eval on first 10
        gt = json.loads(sample['ground_truth'])['gt_parse']

        image = Image.open(sample['image']).convert('RGB')
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = processor.tokenizer(
            TASK_START_TOKEN, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = merged_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=MAX_LENGTH,
                num_beams=3,
                repetition_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
                length_penalty=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.convert_tokens_to_ids(TASK_END_TOKEN),
            )
        inference_time = time.time() - t0
        total_time += inference_time

        raw = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
        raw = raw.replace(TASK_START_TOKEN, "").replace(TASK_END_TOKEN, "")
        raw = raw.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

        try:
            extracted = json.loads(raw)
            json_valid += 1
            valid = True
        except json.JSONDecodeError:
            extracted = {}
            valid = False

        print(f"  Doc {sample['doc_id']}: JSON={'OK' if valid else 'FAIL'}, time={inference_time:.2f}s")
        if valid and extracted:
            for k, v in list(extracted.items())[:2]:
                print(f"    {k}: {str(v)[:80]}")

    print(f"\n  JSON validity: {json_valid}/10")
    print(f"  Avg inference: {total_time/10:.2f}s")
    print(f"\nDone! Model saved to {MERGED_DIR}")


if __name__ == '__main__':
    main()
