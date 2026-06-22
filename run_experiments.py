#!/usr/bin/env python3
"""
Chaos-Eval Experiment Runner
============================

This script runs all experiments with full visibility into:
- What data is being processed
- What prompts are being sent
- What responses are received
- How evaluation is performed
- What metrics are calculated

Run with: python3 run_experiments.py
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
import anthropic

from src.data_loader import DataLoader, Document
from src.prompts import FUNSD_EXTRACTION_PROMPT, CHAOS_EXTRACTION_PROMPT
from src.evaluators import ExtractionEvaluator, EvaluationResult
from src.utils import create_chaos_gradient, add_noise_to_image

# Configuration
MAX_DOCS_PER_TEST = 10  # Limit for quick testing (increase for full run)
CHAOS_LEVELS = 6  # 0-5
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # vision-capable, free tier


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    import io
    buffer = io.BytesIO()
    # Convert to RGB if necessary (for grayscale images)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def extract_with_claude(image: Image.Image, prompt: str, doc_id: str) -> dict:
    """
    Send an image to Claude and get extraction results.

    Returns dict with:
    - raw_response: The text response from Claude
    - parsed_data: Parsed JSON or None
    - success: Whether extraction succeeded
    - latency_ms: Time taken
    - tokens: Token usage
    """
    client = anthropic.Anthropic()

    print(f"    Sending to Claude API...")
    print(f"    Image size: {image.size}")
    print(f"    Prompt length: {len(prompt)} chars")

    start_time = time.time()

    try:
        image_data = image_to_base64(image)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        latency_ms = (time.time() - start_time) * 1000
        raw_response = response.content[0].text

        # Try to parse JSON
        parsed_data = None
        try:
            # Clean up response
            text = raw_response.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                parsed_data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parse error: {e}")

        print(f"    ✓ Response received in {latency_ms:.0f}ms")
        print(f"    Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

        return {
            "raw_response": raw_response,
            "parsed_data": parsed_data,
            "success": parsed_data is not None,
            "latency_ms": latency_ms,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        print(f"    ✗ Error: {e}")
        return {
            "raw_response": "",
            "parsed_data": None,
            "success": False,
            "latency_ms": latency_ms,
            "error": str(e),
        }


def extract_with_gemini(image: Image.Image, prompt: str, doc_id: str) -> dict:
    """
    Drop-in replacement for extract_with_claude that uses Google Gemini.
    Same return contract: raw_response, parsed_data, success, latency_ms, tokens_in/out.
    Vision-capable, runs on the free tier. Key read from GEMINI_API_KEY.
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    print(f"    Sending to Gemini API ({GEMINI_MODEL})...")
    print(f"    Image size: {image.size}")
    print(f"    Prompt length: {len(prompt)} chars")

    img = image.convert("RGB") if image.mode != "RGB" else image
    start_time = time.time()

    # Simple backoff for free-tier rate limits (429/quota)
    last_err = None
    for attempt in range(4):
        try:
            response = model.generate_content([prompt, img])
            latency_ms = (time.time() - start_time) * 1000
            raw_response = response.text or ""

            parsed_data = None
            try:
                text = raw_response.strip()
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    parsed_data = json.loads(text[start:end])
            except json.JSONDecodeError as e:
                print(f"    ⚠️  JSON parse error: {e}")

            usage = getattr(response, "usage_metadata", None)
            tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
            tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0

            print(f"    ✓ Response received in {latency_ms:.0f}ms")
            print(f"    Tokens: {tokens_in} in, {tokens_out} out")

            return {
                "raw_response": raw_response,
                "parsed_data": parsed_data,
                "success": parsed_data is not None,
                "latency_ms": latency_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(s in msg for s in ("429", "quota", "rate", "resource", "exhaust")):
                wait = 8 * (attempt + 1)
                print(f"    ⏳ Rate limited, retrying in {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue
            break

    latency_ms = (time.time() - start_time) * 1000
    print(f"    ✗ Error: {last_err}")
    return {
        "raw_response": "",
        "parsed_data": None,
        "success": False,
        "latency_ms": latency_ms,
        "error": str(last_err),
    }


def evaluate_extraction(extracted: dict | None, ground_truth: dict | None, doc_id: str) -> dict:
    """
    Evaluate extraction against ground truth.

    Shows exactly how evaluation works.
    """
    evaluator = ExtractionEvaluator(fuzzy_threshold=80.0)

    if extracted is None:
        print("    ✗ No extracted data to evaluate")
        return {"field_accuracy": 0, "json_valid": False, "hallucinations": 0}

    if ground_truth is None:
        print("    ✗ No ground truth available")
        return {"field_accuracy": None, "json_valid": True, "hallucinations": None}

    result = evaluator.evaluate(extracted, ground_truth, doc_id, "claude")

    print(f"    Evaluation Results:")
    print(f"      Fields expected: {result.fields_expected}")
    print(f"      Fields extracted: {result.fields_extracted}")
    print(f"      Fields correct: {result.fields_correct}")
    print(f"      Field accuracy: {result.field_accuracy:.1%}")
    print(f"      Hallucinations: {result.hallucinations_detected}")

    if result.hallucination_details:
        print(f"      Hallucination details:")
        for h in result.hallucination_details[:3]:
            print(f"        - {h[:50]}...")

    return result.to_dict()


def run_test_1_baseline_extraction():
    """
    Test 1: Baseline Extraction Accuracy on FUNSD

    This test measures how well Claude extracts fields from noisy scanned forms.
    """
    print_section("TEST 1: Baseline Extraction Accuracy (FUNSD)")

    print("OBJECTIVE: Measure extraction accuracy on real noisy scanned forms")
    print(f"DATASET: FUNSD test set (first {MAX_DOCS_PER_TEST} documents)")
    print("MODEL: Claude claude-sonnet-4-20250514")
    print("PROMPT: FUNSD extraction prompt (extracts questions, answers, key-value pairs)")

    # Load data
    print_subsection("Loading Data")
    loader = DataLoader("data")
    docs = list(loader.load_funsd(split="test"))[:MAX_DOCS_PER_TEST]
    print(f"Loaded {len(docs)} documents")

    # Show the prompt being used
    print_subsection("Extraction Prompt")
    print(FUNSD_EXTRACTION_PROMPT[:500] + "...")

    # Run extractions
    print_subsection("Running Extractions")
    results = []

    for i, doc in enumerate(docs):
        print(f"\n[{i+1}/{len(docs)}] Document: {doc.id}")
        print(f"    Image: {doc.image_path.name}")

        # Show ground truth summary
        gt = doc.ground_truth
        print(f"    Ground Truth: {len(gt.get('question', []))} questions, {len(gt.get('answer', []))} answers")

        # Extract
        img = doc.load_image()
        extraction = extract_with_gemini(img, FUNSD_EXTRACTION_PROMPT, doc.id)

        # Show what was extracted
        if extraction["parsed_data"]:
            ex = extraction["parsed_data"]
            print(f"    Extracted: {len(ex.get('questions', []))} questions, {len(ex.get('answers', []))} answers")

        # Evaluate
        evaluation = evaluate_extraction(extraction["parsed_data"], gt, doc.id)

        results.append({
            "document_id": doc.id,
            "extraction": extraction,
            "evaluation": evaluation,
        })

    # Aggregate results
    print_subsection("Aggregate Results")

    total_accuracy = sum(r["evaluation"]["field_accuracy"] for r in results if r["evaluation"]["field_accuracy"] is not None)
    valid_count = sum(1 for r in results if r["evaluation"]["field_accuracy"] is not None)

    avg_accuracy = total_accuracy / valid_count if valid_count > 0 else 0
    json_valid_rate = sum(1 for r in results if r["extraction"]["success"]) / len(results)
    total_hallucinations = sum(r["evaluation"].get("hallucinations", 0) or 0 for r in results)
    avg_latency = sum(r["extraction"]["latency_ms"] for r in results) / len(results)

    print(f"Documents tested: {len(results)}")
    print(f"Average field accuracy: {avg_accuracy:.1%}")
    print(f"JSON valid rate: {json_valid_rate:.1%}")
    print(f"Total hallucinations: {total_hallucinations}")
    print(f"Average latency: {avg_latency:.0f}ms")

    return results


def run_test_2_chaos_gradient():
    """
    Test 2: Chaos Gradient Analysis

    This test measures how accuracy degrades as document quality decreases.
    """
    print_section("TEST 2: Chaos Gradient Analysis")

    print("OBJECTIVE: Find the 'chaos threshold' - where extraction accuracy collapses")
    print(f"METHOD: Degrade documents through {CHAOS_LEVELS} levels, measure accuracy at each")
    print("DEGRADATION: Gaussian noise + blur + salt/pepper + rotation (combined)")

    # Load a few documents for testing
    print_subsection("Loading Data")
    loader = DataLoader("data")
    docs = list(loader.load_funsd(split="test"))[:5]  # Just 5 for chaos test
    print(f"Using {len(docs)} documents for chaos testing")

    # Show chaos levels
    print_subsection("Chaos Level Definitions")
    print("Level 0: Original (no degradation)")
    print("Level 1: 20% intensity - light noise, slight blur")
    print("Level 2: 40% intensity - moderate noise and blur")
    print("Level 3: 60% intensity - heavy degradation, salt/pepper")
    print("Level 4: 80% intensity - severe degradation, rotation")
    print("Level 5: 100% intensity - maximum chaos")

    # Run chaos gradient tests
    print_subsection("Running Chaos Gradient Tests")
    results = []

    for doc in docs:
        print(f"\n=== Document: {doc.id} ===")

        base_img = doc.load_image()
        gradient = create_chaos_gradient(base_img, num_levels=CHAOS_LEVELS - 1)

        for level, chaos_img in gradient:
            print(f"\n  Chaos Level {level}:")

            # Extract
            extraction = extract_with_gemini(chaos_img, FUNSD_EXTRACTION_PROMPT, f"{doc.id}_chaos{level}")

            # Evaluate
            evaluation = evaluate_extraction(extraction["parsed_data"], doc.ground_truth, doc.id)

            results.append({
                "document_id": doc.id,
                "chaos_level": level,
                "extraction": extraction,
                "evaluation": evaluation,
            })

    # Analyze by chaos level
    print_subsection("Results by Chaos Level")

    for level in range(CHAOS_LEVELS):
        level_results = [r for r in results if r["chaos_level"] == level]
        if not level_results:
            continue

        accuracies = [r["evaluation"]["field_accuracy"] for r in level_results if r["evaluation"]["field_accuracy"] is not None]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0

        bar = "█" * int(avg_acc * 20) + "░" * (20 - int(avg_acc * 20))
        print(f"Level {level}: {bar} {avg_acc:.1%}")

    # Find chaos threshold
    print_subsection("Chaos Threshold Analysis")

    for level in range(CHAOS_LEVELS):
        level_results = [r for r in results if r["chaos_level"] == level]
        accuracies = [r["evaluation"]["field_accuracy"] for r in level_results if r["evaluation"]["field_accuracy"] is not None]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0

        if avg_acc < 0.5:
            print(f"⚠️  CHAOS THRESHOLD FOUND: Level {level}")
            print(f"   Accuracy drops below 50% at chaos level {level}")
            break
    else:
        print("✓ Model maintains >50% accuracy across all chaos levels")

    return results


def run_test_3_detailed_example():
    """
    Test 3: Detailed Example Analysis

    Shows one complete example with full input/output visibility.
    """
    print_section("TEST 3: Detailed Example Analysis")

    print("OBJECTIVE: Show exactly what happens in one extraction")

    # Load one document
    loader = DataLoader("data")
    doc = list(loader.load_funsd(split="test"))[0]

    print_subsection("Input Document")
    print(f"Document ID: {doc.id}")
    print(f"Image path: {doc.image_path}")

    img = doc.load_image()
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")

    print_subsection("Ground Truth")
    gt = doc.ground_truth

    print("Headers:")
    for h in gt.get("header", []):
        print(f"  - {h}")

    print("\nQuestions:")
    for q in gt.get("question", []):
        print(f"  - {q}")

    print("\nAnswers:")
    for a in gt.get("answer", []):
        print(f"  - {a}")

    print("\nKey-Value Pairs:")
    for kv in gt.get("key_value_pairs", []):
        print(f"  - {kv['key']} -> {kv['value']}")

    print_subsection("Prompt Sent to Claude")
    print(FUNSD_EXTRACTION_PROMPT)

    print_subsection("Claude Response")
    extraction = extract_with_gemini(img, FUNSD_EXTRACTION_PROMPT, doc.id)

    print("\nRaw Response:")
    print(extraction["raw_response"][:2000])
    if len(extraction["raw_response"]) > 2000:
        print("... (truncated)")

    print_subsection("Parsed Extraction")
    if extraction["parsed_data"]:
        print(json.dumps(extraction["parsed_data"], indent=2)[:2000])
    else:
        print("Failed to parse JSON")

    print_subsection("Evaluation")
    evaluation = evaluate_extraction(extraction["parsed_data"], gt, doc.id)

    return {
        "document_id": doc.id,
        "ground_truth": gt,
        "extraction": extraction,
        "evaluation": evaluation,
    }


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {filepath}")


def main():
    """Run all experiments."""
    print_section("CHAOS-EVAL EXPERIMENT RUNNER")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Working directory: {Path.cwd()}")

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    print("✓ ANTHROPIC_API_KEY is set")

    all_results = {}

    # Run tests
    try:
        # Test 3 first - detailed example (quick, shows everything)
        detailed_results = run_test_3_detailed_example()
        all_results["detailed_example"] = detailed_results

        # Test 1 - baseline extraction
        baseline_results = run_test_1_baseline_extraction()
        all_results["baseline_extraction"] = baseline_results

        # Test 2 - chaos gradient
        chaos_results = run_test_2_chaos_gradient()
        all_results["chaos_gradient"] = chaos_results

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")

    # Save all results
    print_section("SAVING RESULTS")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"experiment_{timestamp}.json")

    print_section("EXPERIMENT COMPLETE")
    print(f"Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
