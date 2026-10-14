# SemPort

## Model Configuration

All models are configured in `src/module/generator/config.ini`. Fill in the
endpoint and API key before running (fields in the repo are empty placeholders).

- `--model` must be listed in `[model]` → `model_list`.
- Each model gets a section with the same name:
  - **Azure OpenAI** (section contains `azure_endpoint`): fill
    `azure_endpoint` / `azure_deployment` / `azure_api_key` / `azure_api_version`.
  - **OpenAI-compatible**: fill `base_url` / `deployment` / `api_key`.

## Run

CVE IDs come from the `cve_id` column of `Datasets/metadata_linux.csv` /
`Datasets/metadata_android.csv` (`$CVES` is a placeholder, space/comma
separated). Requires Python >= 3.10 with `venv/` at the repo root.

```bash
PYTHONPATH=src venv/bin/python src/main.py run $CVES \
  --mode backport --name semport --model GPT-4o-test \
  --validate-retry 5 --semantic-retry 5 \
  --llm-threads 8 --threads 8
```

Results are written to `results/<mode>/summary_<timestamp>.json`.
