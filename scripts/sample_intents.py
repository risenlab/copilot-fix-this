# Draw a random sample of 361 prompts for intent validation.

import pandas as pd

SAMPLE_SIZE = 361
RANDOM_SEED = 42

dataset = pd.read_csv(
    "data/developer_intents.csv",
    keep_default_na=False,
)

# Sample without replacement and use a fixed seed to make the draw reproducible.
sample = dataset.sample(
    n=SAMPLE_SIZE,
    replace=False,
    random_state=RANDOM_SEED,
)

sample.to_csv("annotation/stratified_sample_361_llm_intents.csv", index=False)