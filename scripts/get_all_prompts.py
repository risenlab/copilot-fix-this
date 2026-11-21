# Export one row per AgentLogs session with its stored prompt and earliest logged problem statement.

import duckdb
from pathlib import Path

source = "hf://datasets/risenlab/agentlogs"
output = Path("replication_package/data/session_prompts.parquet")
temporary_output = output.with_suffix(".parquet.inprogress")

# Extract the first non-empty problem statement from parsed user entries in each session.
query = """
WITH candidates AS (
    SELECT
        l.session.id AS session_id,
        l.entry_index,
        nullif(trim(regexp_extract(
            array_to_string(
                list_transform(l.data.content, lambda block: block.text),
                chr(10)
            ),
            '(?s)<problem_statement>(.*?)</problem_statement>',
            1
        ), chr(9) || chr(10) || chr(13) || ' '), '') AS statement
    FROM read_parquet(?) AS l
    WHERE l.parsed AND l.data.role = 'user'
),
earliest AS (
    SELECT
        session_id,
        min(entry_index) AS log_entry_index,
        arg_min(statement, entry_index) AS log_problem_statement
    FROM candidates
    WHERE statement IS NOT NULL
    GROUP BY session_id
)
SELECT
    s.id AS session_id,
    nullif(s.prompt, '') AS prompt,
    e.log_entry_index,
    e.log_problem_statement
FROM read_parquet(?) AS s
LEFT JOIN earliest AS e ON s.id = e.session_id
"""

with duckdb.connect() as con:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET enable_progress_bar = true")

    con.sql(query, params=[
        f"{source}/agent_session_logs/*.parquet",
        f"{source}/agent_sessions/*.parquet",
    ]).write_parquet(str(temporary_output), compression="zstd")

# Replace the final output only after the Parquet write completes.
temporary_output.replace(output)