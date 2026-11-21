# Reproduce the descriptive counts reported in the preliminary results.

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def main():
    candidates = pd.read_csv(
        DATA_DIR / "vulnerability_prompt_candidates.csv", usecols=["signal_type"]
    )
    print(f"Candidate prompts: {len(candidates):,}")
    print(candidates["signal_type"].value_counts().to_string())

    relevance = pd.read_csv(
        DATA_DIR / "vulnerability_relevant_prompts.csv",
        usecols=["llm_status", "llm_vulnerability_related"],
        dtype=str,
        keep_default_na=False,
    )
    relevant = relevance["llm_status"].eq("success") & (
        relevance["llm_vulnerability_related"].str.strip().str.lower().eq("true")
    )
    print(f"\nRelevant prompts: {relevant.sum():,} ({relevant.mean():.1%})")

    intents = pd.read_csv(
        DATA_DIR / "developer_intents.csv",
        usecols=["session_id", "llm_user_intent"],
        dtype=str,
        keep_default_na=False,
    )
    # Count each intent once per prompt, even if a label appears twice.
    labels = intents["llm_user_intent"].str.split(";").explode().str.strip()
    labels = labels[labels.ne("")].reset_index(name="label").drop_duplicates()
    counts = labels["label"].value_counts()
    print(f"\nIntent counts across {len(intents):,} prompts")
    for label, count in counts.items():
        print(f"{label}: {count:,} ({count / len(intents):.1%})")

    multiple = labels.groupby("index").size().gt(1).sum()
    print(f"Multiple intents: {multiple:,} ({multiple / len(intents):.1%})")

    fix_rows = labels.loc[labels["label"].eq("fix_vulnerability"), "index"]
    fix_ids = intents.loc[fix_rows, ["session_id"]]
    print(f"\nPrompts with a fix intent: {len(fix_ids):,}")

if __name__ == "__main__":
    main()
