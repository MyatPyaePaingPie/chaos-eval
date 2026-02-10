# Building a Specialized Document Extraction Model: Deep Research

**Research Date:** 2026-01-28
**Scope:** Architectures, training, distillation, deployment, datasets, market landscape

---

## Table of Contents

1. [Model Architectures](#1-model-architectures)
2. [OCR-Free vs OCR-Dependent](#2-ocr-free-vs-ocr-dependent)
3. [Knowledge Distillation from LLMs](#3-knowledge-distillation)
4. [Data Augmentation for Documents](#4-data-augmentation)
5. [Training Pipelines](#5-training-pipelines)
6. [Hallucination Prevention](#6-hallucination-prevention)
7. [Cost & Latency Analysis](#7-cost-latency)
8. [Available Datasets](#8-datasets)
9. [Open-Source Tools & Frameworks](#9-open-source-tools)
10. [Market Landscape](#10-market-landscape)
11. [Recommended Path Forward](#11-recommendations)

---

## 1. Model Architectures

### The Key Players

| Model | Params | Type | License | OCR Needed | Best For |
|-------|--------|------|---------|------------|----------|
| **Donut** | 200M | VLM (OCR-free) | MIT | No | Receipt/invoice parsing, JSON output |
| **SmolDocling** | 256M | VLM (OCR-free) | Open | No | Full-page conversion, ultra-low compute |
| **LayoutLM v1** | 113M | Layout-aware | MIT | Yes | Commercial use, entity extraction |
| **LayoutLMv3** | 125M-368M | Layout-aware | CC-BY-NC-4.0 | Yes | Token classification, relation extraction |
| **Florence-2** | 0.2B-0.7B | VLM | MIT | No | Multi-task vision-language |
| **Qwen2.5-VL** | 2B-72B | VLM | Apache 2.0 | No | High-accuracy document QA |
| **MiniCPM-o** | 8B | VLM | Open | No | Tops OCRBench, beats GPT-4o |
| **Pix2Struct** | 282M-1.3B | VLM | Apache 2.0 | No | Charts, tables, UI screenshots |
| **Nougat** | 350M | VLM | CC-BY-NC | No | Academic papers, math notation |
| **TrOCR** | 64M-558M | VLM | MIT | No | Handwriting recognition |
| **UDOP** | 800M | Unified VLM | MIT | No | Multi-task document understanding |
| **docTR** | Varies | OCR engine | Apache 2.0 | N/A | Text detection + recognition |

### Architecture Deep Dives

**Donut (ECCV 2022)** — [Paper](https://arxiv.org/abs/2111.15664) | [Code](https://github.com/clovaai/donut)
- Swin Transformer encoder → BART text decoder
- Pre-trained on SynthDoG (synthetic documents from Wikipedia + ImageNet)
- Includes SynthDoG generator for creating pre-training data in any language
- Think of it as "T5 for documents" — generative, outputs structured text

**SmolDocling (March 2025)** — [Paper](https://arxiv.org/abs/2503.11576) | [Model](https://huggingface.co/ds4sd/SmolDocling-256M-preview)
- 256M params, outperforms models **27x larger** on full-page transcription
- SigLIP vision backbone + SmolLM-2 language backbone
- Introduces DocTags markup format preserving spatial location
- Handles OCR, layout, code, formulas, charts, tables in one model
- From IBM Research + HuggingFace

**LayoutLMv3 (2022)** — [Paper](https://arxiv.org/abs/2204.08387) | [Code](https://github.com/microsoft/unilm)
- Pre-trains with masked language + masked image modeling on text, layout, and image
- 83.37 ANLS on DocVQA, 95% on RVL-CDIP classification
- Think of it as "BERT for documents" — discriminative, outputs per-token labels
- **License warning:** v2/v3 weights may have non-commercial restrictions

**Florence-2 (June 2024)** — [Model](https://huggingface.co/microsoft/Florence-2-large)
- Microsoft's foundation vision-language model
- Pre-trained on FLD-5B (5.4B annotations across 126M images)
- All vision tasks formulated as text generation via prompts
- MIT licensed, commercially usable
- [Fine-tuned for DocVQA](https://huggingface.co/HuggingFaceM4/Florence-2-DocVQA) using only 5% of Docmatix dataset

**Extract-0 (2025)** — [Article](https://ai-engineering-trend.medium.com/a-smaller-model-outperforms-gpt-4-1-on-document-information-extraction-for-196-9e25b39d9538)
- Fine-tuned 7B model that **outperforms GPT-4.1** on structured document extraction
- Training cost: **$196**
- 0.573 average reward score, 89% valid JSON generation rate

---

## 2. OCR-Free vs OCR-Dependent

### Two Paradigms

**OCR-then-Parse Pipeline:**
```
Document Image → OCR Engine → Text + Bounding Boxes → Language Model → Structured Output
```
- Pros: Modular, interpretable, mature OCR engines
- Cons: Error propagation, high compute, language-dependent

**OCR-Free End-to-End:**
```
Document Image → Vision Encoder → Text Decoder → Structured Output
```
- Pros: No error propagation, simpler pipeline, cross-language
- Cons: More training data needed, may struggle with dense small text

### Which to Choose?

```
IF need bounding box locations in output:
  → LayoutLMv3 (needs OCR) or SmolDocling (OCR-free with DocTags)

IF simple JSON extraction (receipts, invoices):
  → Donut (OCR-free, MIT license)

IF complex multi-page documents:
  → Qwen2.5-VL or Florence-2

IF commercial license required:
  → LayoutLM v1 (MIT) or Donut (MIT) or Florence-2 (MIT)

IF ultra-low compute (edge/mobile):
  → SmolDocling (256M) or Donut (200M)
```

---

## 3. Knowledge Distillation from LLMs

### The Core Idea

Use Claude/GPT-4 as a "teacher" to label documents, then train a smaller "student" model. Avoids expensive human annotation.

### Key Papers

**DocKD (EMNLP 2024)** — [Paper](https://aclanthology.org/2024.emnlp-main.185.pdf)
- LLM generates question-answer pairs from document text
- Injects layout knowledge into OCR text before sending to teacher
- Student models require **zero human-labeled annotations**

**Stratos — End-to-End Distillation** — [Paper](https://arxiv.org/html/2510.15992)
- Automatically selects teacher-student pairs
- Incorporates deployment constraints
- When task falls outside teacher's domain, switches to "knowledge injection" mode

**MiniLLM** — [Paper](https://arxiv.org/pdf/2306.08543)
- Minimizes reverse KL divergence instead of forward KLD
- Produces more precise, better-calibrated outputs

### Practical Distillation Pipeline

```
Step 1: Collect 100-500 representative document samples
Step 2: Run through Claude/GPT-4 with detailed extraction prompts
Step 3: Validate teacher outputs (human spot-check ~10-20%)
Step 4: Format as instruction-tuning dataset (image → JSON)
Step 5: Fine-tune student model (Donut, Florence-2, or LayoutLM)
Step 6: Evaluate on held-out test set
Step 7: Iterate on failure cases with targeted data generation
```

### Tools for Distillation

| Tool | Purpose |
|------|---------|
| **DataDreamer** | Synthetic data generation + reproducible LLM workflows (ACL 2024) |
| **Distilabel** | AI Feedback framework for building datasets with LLMs (Argilla) |
| **EasyDistill** | Comprehensive KD toolkit for LLMs (2025) |

---

## 4. Data Augmentation for Documents

Document augmentation differs from general image augmentation — both visual structure AND textual content must stay consistent.

### Specialized Libraries

**Augraphy** — [GitHub](https://github.com/sparkfish/augraphy) | [Paper](https://arxiv.org/abs/2208.14558)
- **Purpose-built for document images**
- Simulates: printing artifacts, scanning noise, fax distortion, ink degradation, coffee stains, paper aging
- Categories: ink effects (bleedthrough, letterpress), paper effects (dithering, folding), noise effects, spatial effects

```python
import augraphy
pipeline = augraphy.default_augraphy_pipeline()
augmented = pipeline(clean_document_image)
```

**SynthDoG** — Part of [Donut repo](https://github.com/clovaai/donut)
- Generates synthetic pre-training data from Wikipedia text + ImageNet backgrounds
- Supports multiple languages, generates millions of samples

**DoGe (DocumentGenerator)** — [GitHub](https://github.com/Travvy88/DocumentGenerator_DoGe)
- Creates document images with headings, tables, paragraphs
- Outputs annotated images with text + bounding boxes
- Integrates with Augraphy for post-generation degradation

**Microsoft Genalog** — [Docs](https://microsoft.github.io/genalog/index.html)
- Generates document images with controllable synthetic noise

### Recommended Augmentation Strategy

```
Tier 1 (Always apply):
  - Gaussian noise (sigma 5-25)
  - JPEG compression artifacts (quality 30-95)
  - Brightness/contrast jitter (±20%)
  - Slight rotation (±2°)

Tier 2 (50% probability):
  - Gaussian blur (kernel 3-7)
  - Shadow simulation
  - Perspective distortion
  - Resolution downscaling (0.5x-0.8x then upscale)

Tier 3 (10-20% probability):
  - Augraphy paper degradation (folding, staining)
  - Ink bleedthrough
  - Partial occlusion
  - Heavy skew (±15°)
```

### Critical Caveat: Model Collapse

Training on synthetic data recursively loses distribution fidelity. **Always mix synthetic with real data.** Gerstgrasser et al. (2024) showed accumulating mixtures of real + synthetic maintain test loss across architectures.

---

## 5. Training Pipelines

### Hardware Requirements

| Model | Minimum GPU | Recommended | Training Time (1K docs) |
|-------|-------------|-------------|------------------------|
| LayoutLM (113M) | T4 16GB | A10G 24GB | 1-3 hours |
| Donut (200M) | T4 16GB | RTX 3090 24GB | 2-5 hours |
| SmolDocling (256M) | T4 16GB | A10G 24GB | 3-6 hours |
| Florence-2 (0.7B) | A10G 24GB | A100 40GB | 4-8 hours |
| Qwen2.5-VL (7B) | A100 40GB | A100 80GB | 8-24 hours |

### Donut Fine-Tuning

Reference: [Phil Schmid's Tutorial](https://www.philschmid.de/fine-tuning-donut)

```python
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    predict_with_generate=True,
)
```

### LayoutLMv3 Fine-Tuning

Reference: [TDS Invoice Tutorial](https://towardsdatascience.com/fine-tuning-layoutlm-v3-for-invoice-processing-e64f8d2c87cf/)

```python
from transformers import (
    LayoutLMForTokenClassification,
    TrainingArguments,
    Trainer
)

model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=num_labels,
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    learning_rate=3e-5,
    fp16=True,
    metric_for_best_model="overall_f1",
)
```

### Hyperparameters Quick Reference

| Parameter | LayoutLM | Donut | 7B VLM (LoRA) |
|-----------|----------|-------|---------------|
| Learning rate | 3e-5 | 2e-5 | 1e-4 |
| Batch size | 16 | 2 | 1-2 |
| Epochs | 15-100 | 3-5 | 1-3 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Max seq length | 512 | 768-1536 | 2048-4096 |
| FP16 | Yes | Yes | Yes (BF16 for A100) |

---

## 6. Hallucination Prevention

The most critical challenge. A model that fabricates an invoice number is worse than one that returns "not found."

### The Problem

Document models hallucinate in two ways:
1. **Fabricating values** — generating plausible but incorrect field values
2. **Over-confident extraction** — high confidence on wrong extractions

### Multi-Layer Prevention Stack

```
Layer 1 — Training Time:
  - Include "N/A" / "NOT_FOUND" as valid extraction targets
  - Add negative examples (documents missing certain fields)
  - Use calibration-aware loss functions

Layer 2 — Inference Time:
  - Token-level confidence scoring with abstention threshold
  - Multi-pass generation with consistency check
  - Temperature scaling on validation set

Layer 3 — Post-Processing:
  - Format validation (regex for dates, amounts, IDs)
  - Cross-field consistency checks (line items sum to total)
  - Retrieval-augmented verification against source text

Layer 4 — Human-in-the-Loop:
  - Route low-confidence extractions to human review
  - Log and retrain on corrected examples
```

### Implementation Example

```python
def extract_with_confidence(model, image, threshold=0.85):
    outputs = model.generate(
        image, output_scores=True,
        return_dict_in_generate=True
    )
    token_probs = [
        torch.softmax(score, dim=-1).max().item()
        for score in outputs.scores
    ]
    avg_confidence = sum(token_probs) / len(token_probs)

    if avg_confidence < threshold:
        return {
            "value": None,
            "confidence": avg_confidence,
            "status": "NEEDS_REVIEW"
        }

    decoded = processor.decode(outputs.sequences[0])
    return {
        "value": decoded,
        "confidence": avg_confidence,
        "status": "EXTRACTED"
    }
```

### Key Research

- OpenAI (Sep 2025): Next-token training objectives reward confident guessing over calibrated uncertainty — models learn to "bluff"
- This drives research toward **calibration-aware reward schemes** where refusal is a valid outcome
- [Frontiers Survey on LLM Hallucinations](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1622292/full)

---

## 7. Cost & Latency Analysis

### Per-Page Economics

| Approach | Cost/1K Pages | Latency/Page | Accuracy |
|----------|--------------|--------------|----------|
| GPT-4o API | $20-$1,000+ | 2-15s | ~98% |
| GPT-4o-mini API | $2-$50 | 1-5s | ~90-95% |
| Azure Document Intelligence | ~$10 | 0.5-2s | ~95% |
| Self-hosted Qwen2.5-VL (7B) | ~$0.50-$1.50 | 1-3s | ~75% JSON |
| Self-hosted Donut/SmolDocling | ~$0.14-$0.70 | 0.1-1.2s | 80-90% (trained domain) |
| Self-hosted PaddleOCR + rules | ~$0.05-$0.20 | 50-200ms | High on templates |

### Break-Even Analysis

```
 1,000 pages/month  →  API is cheaper (no GPU overhead)
10,000 pages/month  →  Roughly break-even
100,000 pages/month →  Self-hosted is 10-100x cheaper
1M+ pages/month     →  Self-hosted is dramatically cheaper

GPU amortization:
  A100 spot: ~$1.50/hour
  SmolDocling: ~3,000 pages/hour
  = $0.0005/page vs GPT-4o at $0.02-0.10/page
```

### Recommended Hybrid Architecture

```
Document Input
    │
    ▼
[Fast OCR / Layout Model]  ← docTR, PaddleOCR, SmolDocling
    │
    ▼
[Template Matching / Rules] ← handles known document types
    │                    │
    ▼ (high conf)        ▼ (low conf / complex)
[Direct Output]      [LLM Fallback]  ← GPT-4o-mini or local 7B
    │                    │
    ▼                    ▼
[Confidence Check] → [Human Review Queue]
```

This gives sub-second latency and sub-cent costs for 80% of documents, reserving expensive LLM calls for the 20% that need reasoning.

---

## 8. Available Datasets

### Pre-Training

| Dataset | Size | Use |
|---------|------|-----|
| **IIT-CDIP** | ~7M documents | Pre-training (tobacco litigation) |
| **RVL-CDIP** | 400K images, 16 classes | Document classification |

### Information Extraction

| Dataset | Size | Use |
|---------|------|-----|
| **FUNSD** | 199 forms | Form understanding, entity labeling |
| **SROIE** | 1,000 receipts | Receipt key info extraction |
| **CORD** | ~1,000 receipts | Receipt parsing (Donut benchmark) |
| **Form-NLU** | Digital + printed + handwritten | VRD-IU 2024 competition |

### Layout Analysis

| Dataset | Size | Use |
|---------|------|-----|
| **PubLayNet** | 360K+ pages | Layout detection (text, tables, figures) |
| **DocBank** | 500K pages, 12 types | Token classification |
| **TableBank** | Large-scale | Table detection |

### Document QA

| Dataset | Use |
|---------|-----|
| **DocVQA** | Visual QA on documents (main benchmark) |
| **InfographicVQA** | QA on infographics |
| **OmniDocBench** (CVPR 2025) | 1,355 pages, 9 doc types, 4 layouts, 3 languages |

---

## 9. Open-Source Tools & Frameworks

### Document Parsing Frameworks

| Tool | Stars | Speed (sec/page) | Key Feature |
|------|-------|-------------------|-------------|
| **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** | 60K+ | Fast | 100+ languages, edge-deployable |
| **[Docling](https://github.com/docling-project/docling)** (IBM) | High | 3.1 (CPU), 0.49 (GPU) | 97.9% table accuracy, MIT |
| **[MinerU](https://github.com/opendatalab/MinerU)** | High | 3.3 (CPU), 0.21 (GPU) | Fastest GPU, PDF→Markdown |
| **[Surya](https://github.com/datalab-to/surya)** | High | Good | 90+ languages, layout analysis |
| **[Marker](https://github.com/VikParuchuri/marker)** | High | 16+ (CPU) | PDF→Markdown/JSON/HTML |
| **[Unstructured](https://github.com/Unstructured-IO/unstructured)** | High | 4.2 (CPU) | ETL for LLM pipelines |

### Model-Specific Repos

| Repo | What It Does |
|------|-------------|
| **[Donut](https://github.com/clovaai/donut)** | OCR-free extraction + SynthDoG generator |
| **[unilm](https://github.com/microsoft/unilm)** | LayoutLM family implementations |
| **[document-ai-transformers](https://github.com/philschmid/document-ai-transformers)** | HuggingFace examples for doc AI |
| **[OmniDocBench](https://github.com/opendatalab/OmniDocBench)** | CVPR 2025 benchmark suite |

### Edge Deployment

| Framework | Best For |
|-----------|----------|
| ONNX Runtime Mobile | iOS/Android, NNAPI/CoreML |
| ExecuTorch | 50KB runtime, 12+ hardware backends |
| OnnxTR | ONNX-wrapped docTR, 8-bit quantized |
| PaddleOCR Lite | Explicit mobile/embedded/IoT support |

---

## 10. Market Landscape

### The IDP Market

- Valued at ~$1.5B in 2022, projected **$17.8B by 2032** (28.9% CAGR)
- 63% of Fortune 250 companies have implemented IDP
- Financial sector leads adoption at 71%
- 100+ vendors according to Gartner

### Key Startups

| Company | Funding | Approach | Scale |
|---------|---------|----------|-------|
| **Reducto** | $108M (a16z) | OCR + VLMs hybrid | ~1B pages/month |
| **Docugami** | Series A | 10-15 specialized LLMs (2.7B-20B) | Enterprise |
| **Hyperscience** | $100M Series D | Agentic document automation | FedRAMP certified |
| **Rossum** | Funded | Transactional LLM | Supply chain focus |

### Cloud Providers

| Provider | Strength | Weakness |
|----------|----------|----------|
| **Azure Document Intelligence** | Custom model training, best overall accuracy | Complex setup |
| **Google Document AI** | 100+ languages, handwriting in 50+ | Weaker on multi-column tables |
| **AWS Textract** | Fastest, serverless-native | Less flexible custom models |

### Kaggle Competition Insights (VRD-IU 2024)

Winning approach: **YOLO + RT-DETR ensemble** with layout-specific post-editor
- Hierarchical Layout Decomposition
- Combined object detection with text recognition
- Used Augraphy for document augmentation

---

## 11. Recommended Path Forward

### If You Want to Build This

**Phase 1: Baseline (Week 1-2)**
- Use Claude/GPT-4o-mini via API to establish accuracy baseline
- Collect 200-500 representative documents
- Manually verify 50-100 outputs
- Result: target accuracy, field difficulty ranking, initial labeled dataset

**Phase 2: Distillation Dataset (Week 2-4)**
- Use validated LLM outputs as training data
- Apply Augraphy augmentation for 5-10x synthetic variants
- Add "NOT_FOUND" examples for missing fields
- Target: 2,000-10,000 training examples

**Phase 3: Model Training (Week 4-5)**
- For known templates: fine-tune LayoutLM v1 (MIT, 113M)
- For OCR-free: fine-tune Donut (MIT, 200M) or SmolDocling (256M)
- For high accuracy on diverse docs: Florence-2 (MIT, 0.7B)
- Hardware: single T4/A10G GPU is sufficient for <1B params

**Phase 4: Calibration (Week 5-6)**
- Temperature scaling on validation set
- Set abstention threshold (start at 0.90)
- Format validators + cross-field consistency checks
- Human review queue for low-confidence extractions

**Phase 5: Deployment (Week 6-8)**
- Export to ONNX for CPU inference
- Implement hybrid architecture (fast model + LLM fallback)
- Monitor weekly, retrain monthly

### Best Model for Your Use Case

Based on our chaos-eval experiment (FUNSD documents, messy scanned forms):

**Top recommendation: Donut**
- MIT licensed (commercial OK)
- OCR-free (simpler pipeline)
- 200M params (trainable on consumer GPU)
- SynthDoG included (generate your own training data)
- Proven on CORD, SROIE, DocVQA benchmarks

**Runner-up: SmolDocling**
- 256M params, outperforms 7B models
- Handles tables, formulas, code, charts
- From IBM Research + HuggingFace
- Newer, less community tooling

**If you need max accuracy: Florence-2 (0.7B)**
- Microsoft's foundation model
- MIT licensed
- Pre-trained on 5.4B annotations
- More expensive to train/run but more capable

---

## Key Takeaways

1. **You don't need to build from scratch.** Fine-tune Donut or Florence-2 on your document types.

2. **Distillation works.** Extract-0 beat GPT-4.1 for $196 in training costs. Use Claude to label your training data.

3. **Augraphy is essential.** Purpose-built document augmentation generates realistic degradation.

4. **Hallucination prevention is engineering, not magic.** Confidence thresholds + abstention + human review.

5. **The hybrid architecture wins.** Fast model for 80% of cases, LLM fallback for the hard 20%.

6. **67% of production models are under 8B params.** Small, specialized models beat general-purpose giants on focused tasks.

7. **The market is $17.8B by 2032.** There's real demand for this.

---

## Sources

### Papers
- [Donut: OCR-free Document Understanding Transformer (ECCV 2022)](https://arxiv.org/abs/2111.15664)
- [SmolDocling (arXiv, March 2025)](https://arxiv.org/abs/2503.11576)
- [DocKD: Knowledge Distillation from LLMs (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.185.pdf)
- [MiniLLM: Knowledge Distillation (2023)](https://arxiv.org/pdf/2306.08543)
- [Stratos: End-to-End Distillation (2025)](https://arxiv.org/html/2510.15992)
- [Augraphy: Document Augmentation (2022)](https://arxiv.org/abs/2208.14558)
- [Hallucination Detection Survey (Jan 2026)](https://arxiv.org/pdf/2601.09929)
- [TextMonkey: OCR-Free Multimodal Model (2024)](https://arxiv.org/html/2403.04473v2)
- [DocSLM: Small VLM for Documents (2025)](https://arxiv.org/html/2511.11313v1)

### Tutorials
- [Fine-tuning Donut (Phil Schmid)](https://www.philschmid.de/fine-tuning-donut)
- [Fine-tuning LayoutLM (Phil Schmid)](https://www.philschmid.de/fine-tuning-layoutlm)
- [Fine-tuning Florence-2 (HuggingFace)](https://huggingface.co/blog/finetune-florence2)
- [LayoutLMv3 Invoice Processing (TDS)](https://towardsdatascience.com/fine-tuning-layoutlm-v3-for-invoice-processing-e64f8d2c87cf/)
- [HuggingFace Document AI Blog](https://huggingface.co/blog/document-ai)
- [HuggingFace Document Augmentation](https://huggingface.co/blog/doc_aug_hf_alb)

### Tools & Repos
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 60K+ stars
- [Docling](https://github.com/docling-project/docling) — IBM, MIT license
- [MinerU](https://github.com/opendatalab/MinerU) — Fastest GPU
- [Augraphy](https://github.com/sparkfish/augraphy) — Document augmentation
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — CVPR 2025 benchmark
- [OnnxTR](https://github.com/felixdittrich92/OnnxTR) — Edge deployment

### Market & Benchmarks
- [IDP Market Report 2025 (Docsumo)](https://www.docsumo.com/blogs/intelligent-document-processing/intelligent-document-processing-market-report-2025)
- [LLM vs OCR Cost Comparison (Mindee)](https://www.mindee.com/blog/llm-vs-ocr-api-cost-comparison)
- [Stanford AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report)
- [State of LLMs 2025 (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [Extract-0: Beat GPT-4.1 for $196](https://ai-engineering-trend.medium.com/a-smaller-model-outperforms-gpt-4-1-on-document-information-extraction-for-196-9e25b39d9538)

---

*Compiled from research across arXiv, HuggingFace, GitHub, academic papers, and industry reports*
*2026-01-28*
