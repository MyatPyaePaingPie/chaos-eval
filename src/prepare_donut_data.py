"""
Prepare FUNSD dataset for Donut fine-tuning.

Extracts structured key-value pairs from FUNSD annotations
by following the question→answer linking. Outputs in Donut's
expected format: {"image": path, "ground_truth": json_string}
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def parse_funsd_annotation(annotation_path: str) -> dict:
    """
    Parse a FUNSD annotation file into structured key-value pairs.

    FUNSD annotations contain elements with:
    - text: the OCR'd text
    - label: question | answer | header | other
    - linking: [[from_id, to_id], ...] connecting questions to answers
    - id: unique element identifier

    We follow the linking to create question→answer pairs.
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    elements = {item['id']: item for item in data['form']}

    # Build link map: question_id → set of answer_ids (deduplicated)
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

    # Extract key-value pairs
    kv_pairs = {}
    for q_id, a_ids in link_map.items():
        question = elements[q_id]
        q_text = question['text'].strip().rstrip(':').strip()

        if not q_text or q_text == ':':
            continue

        answers = []
        for a_id in a_ids:
            answer = elements[a_id]
            a_text = answer['text'].strip()
            if a_text:
                answers.append(a_text)

        if answers:
            # Join multiple answers with semicolon
            kv_pairs[q_text] = '; '.join(answers) if len(answers) > 1 else answers[0]

    # Also extract headers as context
    headers = []
    for item in data['form']:
        if item['label'] == 'header' and item['text'].strip():
            headers.append(item['text'].strip())

    result = {}
    if headers:
        result['_headers'] = headers
    result.update(kv_pairs)

    return result


def prepare_dataset(data_dir: str, split: str = 'training') -> list:
    """
    Prepare all documents in a FUNSD split for Donut training.

    Returns list of {"image": relative_path, "ground_truth": json_string}
    """
    split_dir = 'training_data' if split == 'training' else 'testing_data'
    ann_dir = Path(data_dir) / 'dataset' / split_dir / 'annotations'
    img_dir = Path(data_dir) / 'dataset' / split_dir / 'images'

    dataset = []
    errors = []

    for ann_file in sorted(ann_dir.glob('*.json')):
        doc_id = ann_file.stem
        img_file = img_dir / f'{doc_id}.png'

        if not img_file.exists():
            errors.append(f"Image not found: {img_file}")
            continue

        try:
            kv_pairs = parse_funsd_annotation(str(ann_file))

            # Donut expects ground_truth as a JSON string with a task token
            gt = {
                "gt_parse": kv_pairs
            }

            dataset.append({
                "image": str(img_file),
                "ground_truth": json.dumps(gt, ensure_ascii=False),
                "doc_id": doc_id,
                "num_fields": len([k for k in kv_pairs if not k.startswith('_')])
            })
        except Exception as e:
            errors.append(f"Error parsing {ann_file}: {e}")

    return dataset, errors


def print_stats(dataset: list, split: str):
    """Print dataset statistics."""
    total_fields = sum(d['num_fields'] for d in dataset)
    field_counts = [d['num_fields'] for d in dataset]

    print(f"\n{'='*60}")
    print(f"  FUNSD {split} set - Donut Format")
    print(f"{'='*60}")
    print(f"  Documents:     {len(dataset)}")
    print(f"  Total fields:  {total_fields}")
    print(f"  Avg fields:    {total_fields/len(dataset):.1f}")
    print(f"  Min fields:    {min(field_counts)}")
    print(f"  Max fields:    {max(field_counts)}")
    print(f"  Empty (0):     {sum(1 for c in field_counts if c == 0)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    funsd_dir = Path(__file__).parent.parent / 'data' / 'funsd'
    output_dir = Path(__file__).parent.parent / 'data' / 'donut_format'
    output_dir.mkdir(exist_ok=True)

    # Process both splits
    for split in ['training', 'testing']:
        print(f"\nProcessing {split} split...")
        dataset, errors = prepare_dataset(str(funsd_dir), split)

        if errors:
            print(f"  Warnings: {len(errors)}")
            for e in errors[:5]:
                print(f"    - {e}")

        print_stats(dataset, split)

        # Save dataset
        output_file = output_dir / f'{split}.json'
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {output_file}")

        # Print sample
        if dataset:
            sample = dataset[0]
            gt = json.loads(sample['ground_truth'])
            print(f"\n  Sample ({sample['doc_id']}):")
            for k, v in list(gt['gt_parse'].items())[:5]:
                print(f"    {k}: {v}")
            if len(gt['gt_parse']) > 5:
                print(f"    ... ({len(gt['gt_parse'])-5} more fields)")
