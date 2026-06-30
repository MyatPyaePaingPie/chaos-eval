# chaos-eval

Document-AI benchmark. How well do LLMs extract structured data from noisy
scanned forms (FUNSD), and where do they break under added chaos (noise, blur)?

## Stack
- Python. Claude vision extractor via the Anthropic API or Amazon Bedrock.
- Notebooks `01`..`04` run the pipeline. `src/` holds extractors, data_loader,
  evaluators, utils.

## Run
- Keys go in `.env` (gitignored): `ANTHROPIC_API_KEY`, or the Bedrock vars
  (`AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION`, `BEDROCK_CLIENT`, `BEDROCK_MODEL_ID`).
- The Bedrock path activates when `AWS_BEARER_TOKEN_BEDROCK` or `USE_BEDROCK` is set.

## Conventions
- Always `open()` text files with `encoding="utf-8"`. Windows defaults to cp1252
  and crashes on UTF-8 bytes.
- Record new result sets alongside the old ones. Do not overwrite. See `FINDINGS_*.md`.

## Known caveat (eval design)
- Manual review scored about 98.5%. The automated harness scored far lower
  (about 11%) due to strict exact-key matching and over-eager hallucination flags.
  The metric, not the model, is the ceiling. Fixing the evaluator is open work.
