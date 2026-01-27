"""Evaluation metrics for extraction quality."""

import json
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz
from jiwer import wer, cer


@dataclass
class EvaluationResult:
    """Detailed evaluation of an extraction."""
    document_id: str
    model: str

    # Accuracy metrics
    exact_match: bool = False
    fuzzy_score: float = 0.0  # 0-100
    character_error_rate: float = 1.0  # 0-1, lower is better
    word_error_rate: float = 1.0  # 0-1, lower is better

    # Field-level metrics
    fields_expected: int = 0
    fields_extracted: int = 0
    fields_correct: int = 0
    field_accuracy: float = 0.0

    # Hallucination detection
    hallucinations_detected: int = 0
    hallucination_details: list[str] = field(default_factory=list)

    # Quality flags
    json_valid: bool = False
    extraction_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "model": self.model,
            "exact_match": self.exact_match,
            "fuzzy_score": self.fuzzy_score,
            "cer": self.character_error_rate,
            "wer": self.word_error_rate,
            "fields_expected": self.fields_expected,
            "fields_extracted": self.fields_extracted,
            "fields_correct": self.fields_correct,
            "field_accuracy": self.field_accuracy,
            "hallucinations": self.hallucinations_detected,
            "hallucination_details": self.hallucination_details,
            "json_valid": self.json_valid,
            "extraction_complete": self.extraction_complete,
        }


class ExtractionEvaluator:
    """Evaluate extraction quality against ground truth."""

    def __init__(self, fuzzy_threshold: float = 85.0):
        """
        Args:
            fuzzy_threshold: Minimum fuzzy match score to consider correct (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold

    def evaluate(
        self,
        extracted: dict | None,
        ground_truth: dict | None,
        document_id: str = "",
        model: str = ""
    ) -> EvaluationResult:
        """
        Evaluate extracted data against ground truth.

        Args:
            extracted: The extracted data (parsed JSON)
            ground_truth: The expected data
            document_id: Document identifier
            model: Model name
        """
        result = EvaluationResult(document_id=document_id, model=model)

        # Check JSON validity
        result.json_valid = extracted is not None

        if extracted is None or ground_truth is None:
            return result

        # Compare based on ground truth structure
        if isinstance(ground_truth, dict):
            result = self._evaluate_dict(extracted, ground_truth, result)

        return result

    def _evaluate_dict(
        self,
        extracted: dict,
        ground_truth: dict,
        result: EvaluationResult
    ) -> EvaluationResult:
        """Evaluate dictionary extractions."""

        # Flatten both dicts for comparison
        gt_flat = self._flatten_dict(ground_truth)
        ex_flat = self._flatten_dict(extracted)

        result.fields_expected = len([v for v in gt_flat.values() if v])
        result.fields_extracted = len([v for v in ex_flat.values() if v])

        # Compare fields
        correct = 0
        hallucinations = []

        for key, gt_value in gt_flat.items():
            if not gt_value:  # Skip empty ground truth fields
                continue

            ex_value = ex_flat.get(key, "")

            if not ex_value:
                continue

            # Normalize for comparison
            gt_str = str(gt_value).strip().lower()
            ex_str = str(ex_value).strip().lower()

            # Check exact match
            if gt_str == ex_str:
                correct += 1
            else:
                # Check fuzzy match
                score = fuzz.ratio(gt_str, ex_str)
                if score >= self.fuzzy_threshold:
                    correct += 1

        # Check for hallucinations (extracted fields not in ground truth)
        for key, ex_value in ex_flat.items():
            if not ex_value:
                continue

            # Skip metadata fields
            if key in ["confidence", "issues", "quality_assessment"]:
                continue

            gt_value = gt_flat.get(key, "")

            # If extracted but not in ground truth, might be hallucination
            if ex_value and not gt_value:
                # Check if value appears anywhere in ground truth
                gt_all_values = " ".join(str(v) for v in gt_flat.values() if v).lower()
                ex_str = str(ex_value).strip().lower()

                if ex_str and ex_str not in gt_all_values:
                    # Could be hallucination - flag for review
                    if len(ex_str) > 3:  # Ignore very short strings
                        hallucinations.append(f"{key}: {ex_value}")

        result.fields_correct = correct
        result.field_accuracy = correct / result.fields_expected if result.fields_expected > 0 else 0
        result.exact_match = result.field_accuracy == 1.0
        result.hallucinations_detected = len(hallucinations)
        result.hallucination_details = hallucinations
        result.extraction_complete = result.fields_extracted >= result.fields_expected

        # Calculate fuzzy score
        if result.fields_expected > 0:
            result.fuzzy_score = (correct / result.fields_expected) * 100

        # Calculate CER/WER on text fields
        gt_text = self._extract_text(ground_truth)
        ex_text = self._extract_text(extracted)

        if gt_text and ex_text:
            try:
                result.character_error_rate = cer(gt_text, ex_text)
                result.word_error_rate = wer(gt_text, ex_text)
            except Exception:
                pass

        return result

    def _flatten_dict(self, d: dict, parent_key: str = "") -> dict:
        """Flatten nested dictionary."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v
        return items

    def _extract_text(self, d: dict) -> str:
        """Extract all text values from a dict."""
        texts = []

        def recurse(obj):
            if isinstance(obj, str):
                texts.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    recurse(v)
            elif isinstance(obj, list):
                for item in obj:
                    recurse(item)

        recurse(d)
        return " ".join(texts)

    def evaluate_batch(
        self,
        results: list[tuple[dict | None, dict | None, str, str]]
    ) -> list[EvaluationResult]:
        """Evaluate multiple extractions."""
        return [
            self.evaluate(extracted, ground_truth, doc_id, model)
            for extracted, ground_truth, doc_id, model in results
        ]


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple documents."""
    model: str
    dataset: str
    num_documents: int = 0

    # Averages
    avg_field_accuracy: float = 0.0
    avg_fuzzy_score: float = 0.0
    avg_cer: float = 0.0
    avg_wer: float = 0.0

    # Counts
    exact_matches: int = 0
    json_valid_count: int = 0
    complete_extractions: int = 0
    total_hallucinations: int = 0

    # Rates
    exact_match_rate: float = 0.0
    json_valid_rate: float = 0.0
    completion_rate: float = 0.0
    hallucination_rate: float = 0.0  # docs with hallucinations / total docs

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "num_documents": self.num_documents,
            "avg_field_accuracy": round(self.avg_field_accuracy, 4),
            "avg_fuzzy_score": round(self.avg_fuzzy_score, 2),
            "avg_cer": round(self.avg_cer, 4),
            "avg_wer": round(self.avg_wer, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "json_valid_rate": round(self.json_valid_rate, 4),
            "completion_rate": round(self.completion_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "total_hallucinations": self.total_hallucinations,
        }


def aggregate_results(
    evaluations: list[EvaluationResult],
    model: str = "",
    dataset: str = ""
) -> AggregateMetrics:
    """Aggregate evaluation results into summary metrics."""
    if not evaluations:
        return AggregateMetrics(model=model, dataset=dataset)

    n = len(evaluations)
    metrics = AggregateMetrics(
        model=model or evaluations[0].model,
        dataset=dataset,
        num_documents=n
    )

    # Sum up values
    total_accuracy = 0
    total_fuzzy = 0
    total_cer = 0
    total_wer = 0

    for e in evaluations:
        total_accuracy += e.field_accuracy
        total_fuzzy += e.fuzzy_score
        total_cer += e.character_error_rate
        total_wer += e.word_error_rate

        if e.exact_match:
            metrics.exact_matches += 1
        if e.json_valid:
            metrics.json_valid_count += 1
        if e.extraction_complete:
            metrics.complete_extractions += 1
        if e.hallucinations_detected > 0:
            metrics.total_hallucinations += e.hallucinations_detected

    # Calculate averages
    metrics.avg_field_accuracy = total_accuracy / n
    metrics.avg_fuzzy_score = total_fuzzy / n
    metrics.avg_cer = total_cer / n
    metrics.avg_wer = total_wer / n

    # Calculate rates
    metrics.exact_match_rate = metrics.exact_matches / n
    metrics.json_valid_rate = metrics.json_valid_count / n
    metrics.completion_rate = metrics.complete_extractions / n

    docs_with_hallucinations = sum(1 for e in evaluations if e.hallucinations_detected > 0)
    metrics.hallucination_rate = docs_with_hallucinations / n

    return metrics
