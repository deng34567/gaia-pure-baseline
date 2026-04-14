# GAIA Pure Single-Model Weak Baseline

This repository is a deliberately weak GAIA baseline:

- single model only
- one model call per question
- no browsing
- no file access
- no code execution
- no tools
- no multi-agent orchestration

It is intended to serve as a lower-bound comparison against a stronger `ChatDev`-style architecture.

## What It Reuses

This baseline aligns with the existing `ChatDev-global` implementation in two places only:

- GAIA dataset layout and split handling
- GAIA answer normalization and exact-match style scoring

It does **not** depend on `ChatDev-global` at runtime.

## Files

- `run_eval.py`: CLI entrypoint
- `gaia_dataset.py`: GAIA split loading and task formatting
- `gaia_scoring.py`: answer extraction and scoring
- `openai_client.py`: OpenAI-compatible client for vLLM
- `prompts.py`: fixed prompt template
- `reporting.py`: JSON/JSONL output helpers
- `config.example.yaml`: config template
- `scripts/start_vllm_example.sh`: example vLLM startup command

## Install

```bash
python -m pip install -r requirements.txt
```

## Configure

Copy `config.example.yaml` to `config.yaml` and update:

- `model_name`
- `base_url`
- `api_key`
- `gaia_data_dir`
- `output_dir`

## Run

```bash
python run_eval.py \
  --config config.yaml \
  --gaia-data-dir /path/to/GAIA \
  --split validation \
  --level 1 \
  --limit 50 \
  --output-dir results/qwen35_9b_l1
```

## Outputs

The runner writes:

- `predictions.jsonl`
- `summary.json`
- `metrics_by_level.json`
- `metrics_by_attachment_presence.json`

Each prediction record contains:

- `id`
- `level`
- `question`
- `has_attachment`
- `raw_response`
- `pred`
- `answer`
- `correct`

## Important Baseline Constraint

If a GAIA question includes an attachment, this baseline **ignores it**. The prompt only adds a warning that attachments and tools are unavailable. This is intentional and should be described in any comparison as:

`Pure single-model baseline (question-only, no tools, no file access, no browsing)`
