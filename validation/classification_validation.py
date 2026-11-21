# Compare human and model labels for relevance and intent classification.

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score

relevance_df = pd.read_csv("data/relevance_classification_validation.csv", index_col="session_id")

human_df = relevance_df["human_label"].astype(bool)
llm_df = relevance_df["llm_label"].astype(bool)

results_df = pd.DataFrame(
    [
        {
            "n": len(relevance_df),
            "human_positive": human_df.sum(),
            "llm_positive": llm_df.sum(),
            "kappa": cohen_kappa_score(human_df, llm_df),
            "accuracy": accuracy_score(human_df, llm_df),
            "precision": precision_score(human_df, llm_df, zero_division=0),
            "recall": recall_score(human_df, llm_df, zero_division=0),
        }
    ],
    index=["relevant"],
).round(3)
results_df

intent_df = pd.read_csv("data/intent_classification_validation.csv", index_col="session_id")

# Split intent labels while keeping the session ID for each label.
human_labels_df = (
    intent_df["human_labels"]
        .str.split(";").explode()
        .str.strip().dropna().loc[lambda labels: labels.ne("")]
)
llm_labels_df = (
    intent_df["llm_labels"]
        .str.split(";").explode()
        .str.strip().dropna().loc[lambda labels: labels.ne("")]
)
labels = (set(human_labels_df) | set(llm_labels_df))

result_rows = []

# Evaluate each intent separately by checking whether each rater assigned it to the session.
for label in labels:
    human_has = intent_df.index.isin(human_labels_df[human_labels_df == label].index)
    llm_has = intent_df.index.isin(llm_labels_df[llm_labels_df == label].index)
    result_rows.append(
        {
            "label": label,
            "human_positive": human_has.sum(),
            "llm_positive": llm_has.sum(),
            "kappa": cohen_kappa_score(human_has, llm_has),
            "accuracy": accuracy_score(human_has, llm_has),
            "precision": precision_score(human_has, llm_has, zero_division=0),
            "recall": recall_score(human_has, llm_has, zero_division=0),
        }
    )
results_df = pd.DataFrame(result_rows).set_index("label").astype(float)

# Average across intent labels, excluding unrelated from the mean.
results_df.loc["mean"] = results_df.drop(index="unrelated")[[
    "human_positive",
    "llm_positive",
    "kappa",
    "accuracy",
    "precision",
    "recall"
]].mean()
print(results_df.round(3))