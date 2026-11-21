# Pair model predictions with human annotations for relevance and intent validation.

import pandas as pd

llm_df = pd.read_csv("annotation/stratified_sample_375_llm_relevance.csv").drop(
    columns=["request_text", "full_prompt", "task_url", "human_label", "human_note"]
)
human_df = pd.read_csv("annotation/stratified_sample_375_human_relevance.csv")[
    ["session_id", "human_label", "human_note"]
]

# Match relevance labels by session ID and check that every row has exactly one match.
relevance_df = llm_df.merge(human_df, on="session_id", how="inner", validate="one_to_one")
assert len(relevance_df) == len(llm_df) == len(human_df)

relevance_df = relevance_df.rename(
    columns={
        "llm_vulnerability_related": "llm_label",
    }
)

# Convert the human Yes/No labels to booleans for evaluation.
relevance_df["human_label"] = relevance_df["human_label"].map({"Yes": True, "No": False})
assert relevance_df["human_label"].notna().all()

relevance_df.to_csv("data/relevance_classification_validation.csv", index=False)
relevance_df

llm_df = pd.read_csv("annotation/stratified_sample_361_llm_intents.csv").drop(
    columns=["Unnamed: 0", "__row_id", "request_text", "full_prompt"]
)
human_df = pd.read_csv("annotation/stratified_sample_361_human_intents.csv")[
    ["session_id", "human_intent", "human_intent_note"]
]

# Match intent labels by session ID and preserve the human annotation notes.
intent_df = llm_df.merge(human_df, on="session_id", how="inner", validate="one_to_one")
assert len(intent_df) == len(llm_df) == len(human_df)

intent_df = intent_df.rename(
    columns={
        "llm_user_intent": "llm_labels",
        "human_intent": "human_labels",
        "human_intent_note": "human_note",
    }
)

intent_df.to_csv("data/intent_classification_validation.csv", index=False)
intent_df