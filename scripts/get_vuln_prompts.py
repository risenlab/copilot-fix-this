# Extract developer requests from session records and save those with vulnerability signals.

import csv, re, duckdb
from pathlib import Path
from tqdm.auto import tqdm


INPUT_PATH = Path("data/session_prompts.parquet")
BATCH_SIZE = 10_000

OUTPUT_COLUMNS = (
    "session_id",
    "request_text",
    "full_prompt",
    "vulnerability_ids",
    "vulnerability_keywords",
    "signal_type",
    "matched_fields",
)

SECURITY_TERMS = (
    "CVE", "CWE", "vulnerability", "vulnerable", "exploit", "exploitable",
    "security flaw", "zero-day", "one-day", "hidden vuln", "latent vuln",
    "XSS", "SQLi", "injection", "CSRF", "RCE", "ReDoS", "XXE", "open redirect",
    "buffer overflow", "heap overflow", "off by one",
    "format string", "path traversal", "directory traversal",
    "privilege escalation", "unauthorized", "bypass", "sandbox escape",
    "leak", "data leak", "info leak", "information disclosure",
    "memory disclosure", "malicious", "backdoor", "trojan", "spyware", "exploit kit",
    "shellcode", "hotfix", "patch for CVE", "security patch", "vulnerability fix",
    "fixed vulnerability", "fix exploit",
    "DoS", "denial of service", "crash", "infinite loop", "OOM", "unsafe", "dangling pointer", "use after free",
    "null dereference", "memory corruption", "concurrent access",
    "time of check time of use",
    "sanitize", "sanitization", "unvalidated input", "improper input validation",
    "tainted input", "validate input", "parameter tampering",
)

def build_regexes(terms):
    patterns = []
    for term in terms:
        pattern = re.escape(term).replace(r"\ ", r"[\s\-]+")
        if re.match(r"^\w", term):
            pattern = r"\b" + pattern
        if re.match(r".*\w$", term):
            pattern += r"\b"
        patterns.append((re.compile(pattern, re.IGNORECASE), term))
    return tuple(patterns)

KEYWORD_PATTERNS = build_regexes(SECURITY_TERMS)

def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)

PATTERNS: dict[str, re.Pattern[str]] = {
    "CVE": _p(r"\bCVE-\d{4}-\d{4,}\b"),
    "CWE": _p(r"\bCWE-\d{1,5}\b"),
    "GHSA": _p(r"\bGHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}\b"),
    "GO": _p(r"\bGO-\d{4}-\d{4,}\b"),
    "RUSTSEC": _p(r"\bRUSTSEC-\d{4}-\d{4,}\b"),
    "PYSEC": _p(r"\bPYSEC-\d{4}-\d{1,6}\b"),
    "OSV": _p(r"\bOSV-\d{4}-\d{1,6}\b"),
    "MAL": _p(r"\bMAL-\d{4}-(?:\d{1,7}|ghsa-malware-[a-f0-9]+)\b"),
    "GSD": _p(r"\bGSD-\d{4}-\d{4,7}\b"),
    "HSEC": _p(r"\bHSEC-\d{4}-\d{4}\b"),
    "RSEC": _p(r"\bRSEC-\d{4}-\d{1,4}\b"),
    "JLSEC": _p(r"\bJLSEC-\d{4}-\d{1,4}\b"),
    "OSEC": _p(r"\bOSEC-\d{4}-\d{1,4}\b"),
    "PSF": _p(r"\bPSF-\d{4}-\d{1,4}\b"),
    "BREW": _p(
        r"\bBREW-[a-z0-9@._+-]+-(?:CVE-\d{4}-\d{4,}|GHSA(?:-[a-z0-9]{4}){3})\b"
    ),
    "DRUPAL": _p(r"\b(?:DRUPAL-|SA-)(?:CORE|CONTRIB)-\d{4}-\d{3,4}\b"),
    "ANDROID": _p(r"\b(?:ASB|PUB)-A-\d{6,10}\b"),
    "BITNAMI": _p(r"\bBIT-[a-z0-9][a-z0-9._+-]*-\d{4}-\d{4,}\b"),
    "CLEANSTART": _p(r"\bCLEANSTART-\d{4}-[A-Z]{2}\d{5}\b"),
    "RHSA": _p(r"\bRH[SBE]A-\d{4}:\d{1,6}\b"),
    "ALSA": _p(r"\bAL[SBE]A-\d{4}:\d{1,6}\b"),
    "RLSA": _p(r"\bR[LX]SA-\d{4}:\d{1,6}\b"),
    "DEBIAN": _p(r"\b(?:DSA|DLA|DTSA)-\d{1,5}-\d{1,2}\b"),
    "ELA": _p(r"\bELA-\d{1,5}-\d{1,2}\b"),
    "USN": _p(r"\bUSN-\d{1,5}-\d{1,2}\b"),
    "LSN": _p(r"\bLSN-\d{3,4}-\d{1,2}\b"),
    "SUSE": _p(r"\b(?:openSUSE|SUSE)-[SRFO]U-\d{4}:\d{1,6}(?:-\d{1,2})?\b"),
    "MAGEIA": _p(r"\bMGASA-\d{4}-\d{3,5}\b"),
    "PHOTON": _p(r"\bPHSA-\d{4}-\d{4}\b"),
    "OPENEULER": _p(r"\b(?:OESA|openEuler-SA)-\d{4}-\d{1,5}\b"),
    "AZURELINUX": _p(r"\bAZL-\d{4,6}\b"),
    "CHAINGUARD": _p(r"\bCGA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}\b"),
    "ALPINE": _p(r"\bALPINE-CVE-\d{4}-\d{4,}\b"),
    "CURL": _p(r"\bCURL-CVE-\d{4}-\d{4,}\b"),
    "ALPAQUITA": _p(r"\bBELL-CVE-\d{4}-\d{4,}\b"),
    "EEF": _p(r"\bEEF-CVE-\d{4}-\d{4,}\b"),
    "DEBIAN_CVE": _p(r"\bDEBIAN-CVE-\d{4}-\d{4,}\b"),
    "UBUNTU": _p(r"\bUBUNTU-CVE-\d{4}-\d{4,}\b"),
    "ECHO": _p(r"\bECHO-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}\b"),
    "MINIMOS": _p(r"\bMINI-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}\b"),
    "V8": _p(r"\bV8-\d{4}-\d{1,6}\b"),
    "VCPKG": _p(r"\bVCPKG-\d{4}-\d{1,6}\b"),
    "DHI": _p(r"\bDHI-\d{4}-\d{1,6}\b"),
    "LBSEC": _p(r"\bLBSEC-\d{4}-\d{1,6}\b"),
    "TUXCARE": _p(r"\bCLSA-\d{4}:\d{1,12}\b"),
    "RHLW": _p(r"\bRHLW-\d{4}:\d{1,6}\b"),
    "FREEBSD": _p(r"\bFreeBSD-(?:SA|EN)-\d{2}:\d{2}\.[a-z0-9_.-]+\b"),
}

IDENTIFIER = re.compile(
    "|".join(f"(?:{pattern.pattern})" for pattern in PATTERNS.values()),
    re.IGNORECASE,
)

COMMENT_NEW = re.compile(
    r"^[ \t]*<comment_new>(.*?)</comment_new>",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
COMMENT_METADATA = re.compile(
    r"<(author|comment_id)>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
AGENT_MENTION = re.compile(
    r"^\s*@(?:copilot|claude|codex|anthropic-code-agent|openai-code-agent)"
    r"(?:\[(?:agent|bot)\])?(?=\s|[:,]|$)[\s:,]*",
    re.IGNORECASE,
)
ISSUE = re.compile(
    r"<issue_title>(.*?)</issue_title>\s*"
    r"<issue_description>(.*?)</issue_description>",
    re.IGNORECASE | re.DOTALL,
)
WRAPPER = re.compile(
    r"<(?:problem_statement|issue_title|issue_description|pr_title|"
    r"pr_description|comments|comment_new|comment_old|current_datetime)>",
    re.IGNORECASE,
)

def extract_request(prompt, log):
    prompt = prompt or ""
    log = log or ""

    # Check the longer source first when looking for a new comment or an issue.
    sources = sorted((log, prompt), key=len, reverse=True)

    for text in sources:
        comments = COMMENT_NEW.findall(text)
        if comments:
            request = COMMENT_METADATA.sub("", comments[-1]).strip()
            return AGENT_MENTION.sub("", request).strip(), text

    full_prompt = sources[0]

    if prompt.strip() and not WRAPPER.search(prompt):
        return AGENT_MENTION.sub("", prompt).strip(), full_prompt

    # Prefer the last new comment when present, and remove metadata and the leading agent mention.
    for text in sources:
        issue = ISSUE.search(text)
        if issue:
            request = "\n\n".join(part.strip() for part in issue.groups() if part.strip())
            return request, full_prompt

    if log.strip() and not WRAPPER.search(log):
        return AGENT_MENTION.sub("", log).strip(), full_prompt

    return "", full_prompt

def get_candidate(session_id, prompt, log):
    request, full_prompt = extract_request(prompt, log)
    if not request:
        return None

    ids = {match.group().upper() for match in IDENTIFIER.finditer(request)}

    # Mask matched identifiers before keyword matching so an identifier alone does not count as both signals.
    keyword_text = IDENTIFIER.sub(lambda match: " " * len(match.group()), request)
    keywords = {
        label
        for pattern, label in KEYWORD_PATTERNS
        if pattern.search(keyword_text)
    }

    if not ids and not keywords:
        return None

    if ids and keywords:
        signal_type = "both"
    elif ids:
        signal_type = "explicit"
    else:
        signal_type = "implicit"

    return (
        session_id,
        request,
        full_prompt,
        "; ".join(sorted(ids)),
        "; ".join(sorted(keywords)),
        signal_type,
        "request_text",
    )

def main():
    output_path = Path("data/vulnerability_prompt_candidates.csv")
    temporary_path = output_path.with_suffix(".csv.inprogress")

    with duckdb.connect(config={"threads": 1}) as con:
        (total,) = con.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(INPUT_PATH)],
        ).fetchone()

        sessions = con.execute(
            "SELECT session_id, prompt, log_problem_statement FROM read_parquet(?)",
            [str(INPUT_PATH)],
        )

        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(OUTPUT_COLUMNS)

            # Read sessions in batches to limit memory use.
            with tqdm(total=total, desc="Scanning sessions", unit="session") as progress:
                while rows := sessions.fetchmany(BATCH_SIZE):
                    for row in rows:
                        candidate = get_candidate(*row)
                        if candidate is not None:
                            writer.writerow(candidate)
                    progress.update(len(rows))

    # Replace the final CSV only after the scan completes.
    temporary_path.replace(output_path)

if __name__ == "__main__":
    main()