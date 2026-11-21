# Replication package

This is the replication pacakage for _“@copilot, fix this?”: What Coding Agents Actually Do When Asked to Fix Vulnerabilities_. It contains the scripts, annotations, and saved classifications for our preliminary analysis of developer requests in [AgentLogs v1.0](https://huggingface.co/datasets/risenlab/agentlogs).

## How we built the dataset

We retrieved developer prompts from the `agent_sessions` and `agent_session_logs` Parquet files in AgentLogs v1.0. The script `scripts/get_all_prompts.py` joined the stored session prompt to the earliest non-empty problem statement in the user log entries, when available, and saved one row per session in `data/session_prompts.parquet`.

Next, `scripts/get_vuln_prompts.py` extracted the developer prompt and searched it for vulnerability identifiers and security-related terms. This produced 14,378 candidate requests in `data/vulnerability_prompt_candidates.csv`. We used `scripts/classify_relevance.py` to identify 5,825 vulnerability-related requests, then used `scripts/classify_intents.py` to assign intent labels. The resulting labels are in `data/developer_intents.csv`.

## Reproduce the preliminary results

Set up a Python environment and run:

```
python -m pip install -r requirements.txt
```

`annotation/` contains the human labels, model predictions, and the intent codebook: `annotation/codebook-MSR27.docx`. The vulnerability-relevance sample contains 375 prompts stratified by model relevance label and signal type. The intent sample contains 361 prompts. Use the prepared validation CSVs for the commands above. Rebuilding them requires updating the intent input paths in `validation/prepare_classification_validation.py`.

## Rerun the classification pipeline

Both classifiers use `gemini-3.5-flash-lite` and require `GOOGLE_API_KEY` in a `.env` file.

1.  Run `python replication_package/scripts/get_all_prompts.py` to extract prompts from AgentLogs.
2.  Run `python scripts/get_vuln_prompts.py` and then `python scripts/classify_relevance.py`.
3.  Run `python scripts/classify_intents.py`. The script reads `data/vulnerability_relevant_prompts.csv`, keeps successful classifications marked as vulnerability-related, and saves the intent labels to `data/developer_intents.csv`.