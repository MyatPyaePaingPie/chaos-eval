# Chaos-Eval Test Plan

## Overview

This document describes exactly what we're testing, how, and why.

---

## Phase 1: Environment Setup

### 1.1 Dependencies
- Python 3.10+
- Anthropic SDK (Claude API)
- Ollama (local models)
- PIL/Pillow (image processing)
- Standard ML stack (pandas, numpy, matplotlib)

### 1.2 API Configuration
- ANTHROPIC_API_KEY for Claude
- Ollama running locally for open-source models

---

## Phase 2: Data Acquisition

### 2.1 FUNSD Dataset
**Source**: https://guillaumejaume.github.io/FUNSD/
**What it is**: 199 real scanned forms from various sources
**Ground truth**: JSON annotations with:
- `question`: Field labels (e.g., "Name:", "Date:")
- `answer`: Field values (e.g., "John Smith", "01/15/1990")
- `header`: Document titles, headers
- `other`: Miscellaneous text
- `linking`: Which questions link to which answers

**Why we use it**: Real-world messy forms with verified ground truth

### 2.2 SROIE Dataset
**Source**: ICDAR 2019 Competition
**What it is**: 1,000 scanned receipts
**Ground truth**: 4 key fields per receipt:
- company (store name)
- address
- date
- total

**Why we use it**: Degraded real documents with simple extraction targets

### 2.3 Synthetic Chaos
**What it is**: Original documents with programmatic degradation
**Degradation types**:
- Gaussian noise (simulates scan artifacts)
- Salt & pepper noise (simulates dust/damage)
- Blur (simulates motion/focus issues)
- Rotation (simulates misaligned scans)

**Chaos levels**:
- Level 0: Original (baseline)
- Level 1: 20% intensity degradation
- Level 2: 40% intensity
- Level 3: 60% intensity
- Level 4: 80% intensity
- Level 5: 100% intensity (maximum chaos)

---

## Phase 3: Test Design

### Test 1: Baseline Extraction Accuracy
**Question**: How accurately can each model extract fields from noisy documents?

**Method**:
1. Load FUNSD test set (50 documents)
2. For each document:
   - Send image to model with extraction prompt
   - Parse JSON response
   - Compare extracted fields to ground truth
3. Calculate metrics:
   - Field accuracy (exact + fuzzy match)
   - JSON validity rate
   - Hallucination count

**Models tested**:
- Claude 3.5 Sonnet (API)
- Llama 3.2 Vision (local)
- Tesseract (baseline OCR)

### Test 2: Receipt Extraction
**Question**: Can models extract specific fields from degraded receipts?

**Method**:
1. Load SROIE test set (up to 100 documents)
2. Extract: company, address, date, total
3. Compare to ground truth
4. Calculate per-field accuracy

### Test 3: Chaos Gradient Analysis
**Question**: At what degradation level does extraction fail?

**Method**:
1. Take 10 clean documents from FUNSD
2. Generate 6 versions of each (levels 0-5)
3. Extract from each version
4. Plot accuracy vs chaos level
5. Identify "chaos threshold" (where accuracy < 50%)

### Test 4: Hallucination Detection
**Question**: When models can't read text, do they admit it or fabricate?

**Method**:
1. For each extraction, compare output to ground truth
2. Flag extractions where:
   - Value exists in output but NOT in source
   - Value is structurally plausible but incorrect
3. Categorize:
   - Correct extractions
   - Admitted uncertainty ("unclear", null values)
   - Hallucinated values

### Test 5: Cross-Model Consistency
**Question**: Do different models extract the same information?

**Method**:
1. Run same documents through all models
2. Compare outputs pairwise
3. Calculate agreement rate
4. Identify systematic differences

---

## Phase 4: Metrics

### Primary Metrics
| Metric | Definition | Good Value |
|--------|------------|------------|
| Field Accuracy | % of fields correctly extracted | >80% |
| JSON Valid Rate | % of outputs that parse as JSON | >95% |
| Hallucination Rate | % of docs with fabricated values | <5% |
| CER | Character Error Rate on text | <10% |
| WER | Word Error Rate on text | <15% |

### Derived Metrics
- **Chaos Threshold**: Degradation level where accuracy drops below 50%
- **Graceful Degradation Score**: How smoothly accuracy declines (vs sudden collapse)
- **Confidence Calibration**: Do low-confidence outputs correlate with errors?

---

## Phase 5: Analysis

### Visualizations to Generate
1. **Model comparison bar chart**: Accuracy by model
2. **Chaos tolerance curve**: Accuracy vs degradation level
3. **Heatmap**: Document × Chaos Level accuracy
4. **Scatter plot**: Accuracy vs Latency tradeoff
5. **Confusion examples**: Side-by-side of input → output for interesting cases

### Statistical Tests
- Compare model means with confidence intervals
- Test significance of chaos threshold differences

---

## Phase 6: Execution Order

```
1. Setup environment
2. Download FUNSD dataset
3. Verify data loaded correctly
4. Run Test 1: Baseline extraction (FUNSD)
5. Run Test 2: Receipt extraction (SROIE or samples)
6. Run Test 3: Chaos gradient
7. Run Test 4: Hallucination analysis
8. Run Test 5: Cross-model comparison
9. Generate visualizations
10. Compile findings for article
```

---

## Expected Duration

| Phase | Estimated Time |
|-------|----------------|
| Setup | 5 minutes |
| Data download | 2-5 minutes |
| Test 1 (50 docs × 2 models) | 10-15 minutes |
| Test 2 (100 docs × 2 models) | 15-20 minutes |
| Test 3 (10 docs × 6 levels × 2 models) | 20-30 minutes |
| Analysis | 5 minutes |

Total: ~1 hour for complete run

---

## Success Criteria

The experiment succeeds if we can answer:
1. What is the extraction accuracy for each model?
2. Where is the chaos threshold?
3. How prevalent is hallucination?
4. Which model performs best overall?
