# Can AI Read Your Grandpa's Tax Returns? Testing LLM Limits on Messy Documents

*How well do large language models handle the chaos of real-world legacy data?*

---

## The Problem Nobody Talks About

The IRS has over **1 billion historical documents** sitting in storage, costing $40 million per year just to keep the lights on. The National Archives holds **13.5 billion pages** of records. And that's just the federal government.

Every hospital has boxes of handwritten patient records. Every law firm has filing cabinets of old case files. Every insurance company has decades of claims on microfilm.

This data is messy. Faded. Handwritten. Scanned at bad angles. Formatted for systems that no longer exist.

The promise of AI is that it can make sense of this chaos. But can it actually?

I decided to find out.

---

## The Experiment

I tested leading AI models on their ability to extract structured data from messy, degraded, and unstructured documents.

**The Models:**
- Claude 3.5 Sonnet (Anthropic)
- Llama 3.2 Vision (Meta, running locally)

**The Data:**
- FUNSD: 199 real scanned forms with noise, skew, and handwriting
- SROIE: 1,000 degraded receipts with folds, fading, and damage
- Synthetic chaos: The same documents degraded at 5 increasing levels

**The Questions:**
1. How accurately can AI extract specific fields from messy documents?
2. At what point does document degradation break the model?
3. When the model can't read something, does it admit it—or make things up?

---

## What I Found

### Finding 1: [ACCURACY RESULT]

*[Fill in after running experiments]*

> "Claude achieved X% field accuracy on noisy forms, compared to Y% for local models."

### Finding 2: The Chaos Threshold

*[Fill in after running experiments]*

I applied increasing levels of degradation to the same documents:
- Level 0: Original scan
- Level 1: Light noise
- Level 2: Moderate blur + noise
- Level 3: Heavy degradation
- Level 4: Severe damage
- Level 5: Nearly illegible

![Chaos Tolerance Curve](../results/chart_chaos_tolerance.png)

The accuracy held steady until Level X, then collapsed. This is the **chaos threshold**—the point where the model transitions from "working" to "guessing."

### Finding 3: The Hallucination Problem

*[Fill in after running experiments]*

When models couldn't read text clearly, they:
- Admitted uncertainty: X% of the time
- Made plausible guesses: Y% of the time
- Hallucinated confidently wrong answers: Z% of the time

This last category is the dangerous one. A model that says "I can't read this" is useful. A model that invents data is a liability.

---

## What This Means

### For Government Agencies

The 85-90% accuracy range sounds good until you multiply it by a billion documents. That's 100-150 million potential errors.

But here's the thing: human accuracy on this task isn't 100% either. The question isn't "is AI perfect?" but "is AI better than the alternative?"

### For Anyone Dealing with Legacy Data

1. **Hybrid workflows are essential.** The research is clear: OCR + LLM post-correction beats pure approaches.

2. **Build in confidence thresholds.** Flag extractions below X% confidence for human review.

3. **Test on your actual data.** FUNSD accuracy doesn't predict performance on 1970s tax forms.

### For the AI Industry

Document understanding is a solvable problem, but it's not solved yet. The models are good enough for high-volume, low-stakes extraction. They're not ready for high-stakes decisions without human oversight.

---

## Try It Yourself

The full experiment code is available at: [github.com/YourUsername/chaos-eval](https://github.com/YourUsername/chaos-eval)

To run your own tests:

```bash
git clone https://github.com/YourUsername/chaos-eval
cd chaos-eval
pip install -r requirements.txt
python src/download_data.py
jupyter notebook
```

---

## The Bottom Line

Can AI read your grandpa's tax returns?

**Mostly, yes.** With caveats.

It handles printed text well. Handwriting is harder. Severe degradation breaks it. And when it fails, it sometimes lies about it.

The technology is impressive. But "impressive" and "ready for production" are different things.

The chaos threshold is real. Know where yours is before you bet your data on AI.

---

*[Your name] experiments with AI at the intersection of messy reality and clean abstractions. Follow for more explorations of what works, what doesn't, and what's actually useful.*

---

## Appendix: Methodology Notes

### Datasets Used

| Dataset | Source | Documents | Type |
|---------|--------|-----------|------|
| FUNSD | ICDAR 2019 | 199 | Noisy scanned forms |
| SROIE | ICDAR 2019 | 1,000 | Degraded receipts |
| NOD | Zenodo | 18,504 | Multi-noise levels |

### Metrics

- **Field Accuracy**: Exact or fuzzy match (>80% similarity) of extracted values
- **JSON Validity**: Whether the model output parseable structured data
- **Hallucination Rate**: Extracted values not present in source document
- **CER/WER**: Character/Word Error Rate on full text extraction

### Code

All experiment code, data loaders, and analysis notebooks are available in the linked repository.
