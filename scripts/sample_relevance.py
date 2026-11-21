# Draw a stratified sample of 375 prompts for manual relevance validation.

import pandas as pd

INPUT_FILE = "data/vulnerability_relevant_prompts.csv"
OUTPUT_FILE = "annotation/stratified_sample_375_llm_relevance.csv"
SAMPLE_SIZE = 375
RANDOM_SEED = 42
STRATA = ["llm_vulnerability_related", "signal_type"]

df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

if not df["session_id"].is_unique:
    raise ValueError("Expected one row per session_id.")

for column in STRATA:
    df[column] = df[column].str.strip().str.lower()

eligible = df.loc[
    df["llm_status"].eq("success")
    & df["llm_vulnerability_related"].isin(["true", "false"])
    & df["signal_type"].isin(["explicit", "implicit", "both"])
]

# Allocate places proportionally across relevance labels and signal types.
groups = eligible.groupby(STRATA)
allocation = groups.size().to_frame("population_size")
quota = allocation["population_size"] / len(eligible) * SAMPLE_SIZE
allocation["sample_size"] = quota.astype(int)

# Give leftover places to the strata with the largest fractional remainders.
allocation["remainder"] = quota - allocation["sample_size"]
remaining = SAMPLE_SIZE - allocation["sample_size"].sum()
extra_places = allocation.sort_values("remainder", ascending=False).head(remaining).index
allocation.loc[extra_places, "sample_size"] += 1

sample = pd.concat(
    [
        group.sample(
            n=int(allocation.at[key, "sample_size"]),
            random_state=RANDOM_SEED,
        )
        for key, group in groups
    ],
    ignore_index=True,
)

sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
sample["human_label"] = ""
sample["human_note"] = ""
sample.to_csv(OUTPUT_FILE, index=False)