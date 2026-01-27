# chaos-eval

**Testing LLM capabilities on messy, degraded, and unstructured documents.**

Can AI parse chaos? This experiment measures how well large language models handle the kind of documents that haunt government agencies, hospitals, and filing cabinets everywhere - faded forms, handwritten notes, degraded scans, and formatting nightmares.

## The Question

Government agencies hold billions of pages of legacy documents. The IRS alone has 1 billion+ historical documents. The National Archives has 13.5 billion pages. Most of this is messy, inconsistent, and trapped in formats that predate modern databases.

**Can LLMs reliably extract structured data from unstructured chaos?**

## What We're Testing

| Dimension | What It Means |
|-----------|---------------|
| **Extraction Accuracy** | Does the model pull out the correct fields? |
| **Format Consistency** | Is the output clean, structured JSON every time? |
| **Chaos Tolerance** | At what degradation level does performance collapse? |
| **Hallucination Rate** | Does the model fabricate information when uncertain? |
| **Cross-Model Variance** | How do different models compare? |

## Models Tested

**API Models:**
- Claude 3.5 Sonnet (Anthropic)

**Local Models (via Ollama):**
- Llama 3.2 (Meta)
- Mistral 7B
- Qwen 2.5

## Datasets

| Dataset | Documents | Type | Why |
|---------|-----------|------|-----|
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | 199 forms | Noisy scanned forms | Real messy documents with ground truth |
| [SROIE](https://rrc.cvc.uab.es/?ch=13) | 1,000 receipts | Degraded receipts | Real-world noise, folds, damage |
| [NOD](https://zenodo.org/records/5068735) | 18,504 images | Variable noise levels | Same docs at 44 degradation levels |

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# For local models
# Install Ollama: https://ollama.ai
ollama pull llama3.2
ollama pull mistral
ollama pull qwen2.5
```

### Setup API Keys

```bash
cp .env.example .env
# Edit .env with your Anthropic API key
```

### Download Datasets

```bash
python src/download_data.py
```

### Run Experiments

```bash
# Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# Run extraction experiments
jupyter notebook notebooks/02_extraction_experiments.ipynb

# Analyze chaos tolerance
jupyter notebook notebooks/03_chaos_gradient_test.ipynb

# Generate results
jupyter notebook notebooks/04_results_analysis.ipynb
```

## Project Structure

```
chaos-eval/
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Understand the datasets
│   ├── 02_extraction_experiments.ipynb # Core extraction tests
│   ├── 03_chaos_gradient_test.ipynb   # Degradation analysis
│   └── 04_results_analysis.ipynb      # Charts, tables, insights
├── src/
│   ├── __init__.py
│   ├── download_data.py    # Dataset downloaders
│   ├── data_loader.py      # Load and preprocess datasets
│   ├── extractors.py       # LLM extraction logic
│   ├── evaluators.py       # Accuracy, hallucination metrics
│   ├── prompts.py          # Extraction prompts
│   └── utils.py            # Helpers
├── data/
│   ├── funsd/              # FUNSD dataset
│   ├── sroie/              # SROIE dataset
│   ├── nod/                # Noisy OCR dataset
│   └── samples/            # Sample docs for quick tests
├── results/
│   └── ...                 # Experiment outputs
└── article/
    └── draft.md            # Medium article draft
```

## Key Findings

*Results will be populated after running experiments.*

## Article

The findings from this experiment will be published as a Medium article exploring LLM capabilities for legacy document processing.

## License

MIT

## Acknowledgments

- FUNSD dataset: Guillaume Jaume et al.
- SROIE dataset: ICDAR 2019
- NOD dataset: Hegghammer et al.
