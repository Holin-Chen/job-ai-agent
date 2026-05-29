#!/usr/bin/env python3
"""
Job AI Agent — Scorer + Cover Letter Drafter + Resume Tailor + Job Search
──────────────────────────────────────────────────────────────────────────
Usage:
    python job_agent.py

Requires a .env file with:
    ANTHROPIC_API_KEY=sk-ant-...
    ADZUNA_APP_ID=...          # free at developer.adzuna.com
    ADZUNA_API_KEY=...
"""

import os
import sys
import json
import urllib.request
import urllib.error
import urllib.parse
from html.parser import HTMLParser
import anthropic

def _load_env_file() -> None:
    """Read a .env file next to this script without requiring python-dotenv."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()

_load_env_file()

MODEL = "claude-opus-4-7"

# ── Resume text (injected into every request via prompt caching) ────────────
# The cache_control blocks keep this from being re-billed on repeated calls.
RESUME = """
Holin Yangping Chen
San Francisco Bay Area, California | 614-270-3776 | yangpingchen1996@outlook.com
holin-chen.github.io | scholar.google.com/yangpingchen

PROFESSIONAL SUMMARY
Senior Real-World Evidence (RWE) and Healthcare Research Analyst with 5 years of experience
conducting large-scale observational health research using Medicare, Medicaid, and epidemiologic
survey datasets for federal agencies including the FDA and CDC. Deep expertise in
pharmacoepidemiology, causal inference, vaccine surveillance, and statistical programming at
scale (millions to billions of records) using SAS, R, Python, and SQL.

Experienced in designing reproducible analytical pipelines, survival analysis, propensity score
methodologies, longitudinal modeling, and safety surveillance studies involving millions to
billions of healthcare records. Contributed to FDA drug safety evaluations informing regulatory
decisions and boxed warning updates. Published researcher with expertise spanning vaccine
effectiveness, infectious disease epidemiology, and real-world evidence generation.

PROFESSIONAL EXPERIENCE

The SPHERE Institute (Acumen, LLC) | Burlingame, CA | Jul 2022 – Present
Senior Data and Policy Analyst - Statistical Programmer
- Designed and executed large-scale pharmacoepidemiology and RWE studies for FDA and CDC using
  Medicare FFS, Medicare Advantage, and Medicaid claims data; queried, cleaned, and harmonized
  multi-billion-row datasets using SAS, SQL, R, and Python across vaccine surveillance, drug
  safety, and utilization research.
- Built reproducible end-to-end analytical pipelines for vaccine effectiveness and uptake
  surveillance studies, encompassing automated data ingestion, cohort construction, statistical
  modeling, and reporting; developed parallelized ML workflows that improved causal inference
  model runtime by more than 8×.
- Applied retrospective cohort study — propensity score weighting, IPTW, logistic regression,
  and survival analysis — to evaluate post-market drug safety outcomes from observational claims
  data; contributed to an FDA evaluation of Prolia (denosumab) that was published in JAMA and
  led to an FDA-mandated boxed warning update.
- Led the healthcare cost and utilization analysis by comparing branded, biosimilar, and unbranded
  pharmaceuticals using Medicare reimbursement data; designed data modules and metadata
  infrastructure for CDC COVID-19 and RSV vaccine surveillance official reports.
- Produced and presented research deliverables, technical documentation, and data visualizations
  directly to FDA and CDC clients; provided internal training on SAS, Python, and large-scale
  claims data workflows including code optimization and reproducible research practices.

Rollins School of Public Health, Emory University | Atlanta, GA | Jul 2021 – Jul 2022
Public Health Program Associate – Data Manager
- Developed automated data cleaning, QC, and reporting pipelines using R, APIs, and GitHub for
  large-scale epidemiologic survey and sensor-based studies; managed longitudinal diary datasets
  to quantify social contact patterns and disease transmission parameters.
- Built Bayesian MCMC modeling workflows in R for longitudinal COVID-19 serology analyses across
  multiple U.S. states, executed on Linux-based HPC clusters; analyzed wearable sensor and
  household contact network data in Python to evaluate transmission dynamics.
- Generated reproducible weekly stakeholder reports featuring statistical summaries, interactive
  visualizations, and R Markdown deliverables.
- Trained international field teams in Guatemala and India on sensor deployment and data collection
  for global social mixing studies.

EDUCATION
Master of Public Health | Epidemiology | GPA: 3.91/4.0 | 2021
Emory University, Rollins School of Public Health | Atlanta, GA

Bachelor of Medicine | Medical Sciences | 2019
Southern Medical University | Guangzhou, China
""".strip()


# ── Shared system blocks (cached) ───────────────────────────────────────────

def _base_system(role_instruction: str) -> list[dict]:
    """
    Returns two system blocks: a role instruction (cached) and the resume (cached).
    Both blocks carry cache_control so they are written to the prompt cache on the
    first call and served from cache on every subsequent call — no re-billing.
    """
    return [
        {
            "type": "text",
            "text": role_instruction,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"Here is the candidate's resume:\n\n{RESUME}",
            "cache_control": {"type": "ephemeral"},
        },
    ]


# ── Scorer ──────────────────────────────────────────────────────────────────

def score_job(client: anthropic.Anthropic, job_description: str) -> None:
    """
    Uses adaptive thinking so the model can reason deeply before it scores.
    Thinking blocks are hidden from output (Opus 4.7 omits them by default).
    """
    print("\nAnalyzing fit (may take 20-40 s with adaptive thinking)...\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_base_system(
            "You are an expert career coach and recruiter. "
            "Analyze how well a candidate's resume matches a job description. "
            "Be honest, specific, and avoid generic advice."
        ),
        messages=[
            {
                "role": "user",
                "content": f"""Score this job description against the resume.

JOB DESCRIPTION:
{job_description}

Return your analysis in exactly this format:

## Fit Score: [0–100] — [one-line verdict]

## Strengths (what you have that they want)
- ...

## Gaps (what's missing or weak)
- ...

## Resume Tweaks (how to reframe existing bullets to match this JD)
- ...

## Recommendation
[2–3 sentences: should they apply, and how to position themselves?]""",
            }
        ],
    )

    # Opus 4.7 omits thinking text by default — only print text blocks
    for block in response.content:
        if block.type == "text":
            print(block.text)

    _print_cache_stats(response.usage)


# ── Drafter ─────────────────────────────────────────────────────────────────

def draft_cover_letter(client: anthropic.Anthropic, job_description: str) -> None:
    """
    Streams the cover letter so you see it appear word-by-word.
    No thinking block needed here — cover letter writing is generative, not analytical.
    """
    print("\nDrafting cover letter...\n")
    print("─" * 60)

    with client.messages.stream(
        model=MODEL,
        max_tokens=1200,
        system=_base_system(
            "You are an expert cover letter writer. "
            "Write compelling, specific cover letters tailored to the job. "
            "Never use generic filler. Keep it under 400 words. "
            "Tone: professional but direct, confident without being arrogant."
        ),
        messages=[
            {
                "role": "user",
                "content": f"""Write a cover letter for this job.

JOB DESCRIPTION:
{job_description}

Requirements:
- Open with "Dear Hiring Manager," — no company name placeholders
- Hook: first sentence shows you understand what the role is actually solving for
- Body: connect 2–3 concrete achievements from the resume to the role's key needs
- Close: short, confident call to action — no hollow phrases like "I look forward to hearing from you"
- Sign off as "Holin Chen"
- No clichés (no "excited to apply", no "passion for", no "team player")""",
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n" + "─" * 60)
    _print_cache_stats(stream.get_final_message().usage)


# ── Resume Tailor ────────────────────────────────────────────────────────────

def tailor_resume(client: anthropic.Anthropic, job_description: str) -> None:
    """
    Reads Resume.tex, rewrites the highest-impact bullets for the JD,
    shows a clean diff (ORIGINAL → REWRITTEN), then saves Resume_tailored.tex.
    Uses adaptive thinking to reason about which changes matter most.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, "Resume.tex")

    if not os.path.exists(tex_path):
        print("\nError: Resume.tex not found next to job_agent.py.")
        return

    with open(tex_path, encoding="utf-8") as f:
        resume_tex = f.read()

    print("\nTailoring resume to this job (may take 30-50 s)...\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": (
                    "You are an expert resume writer and ATS optimization specialist. "
                    "You rewrite resume bullets to match job descriptions precisely — "
                    "mirroring the JD's exact keywords, verbs, and framing — while "
                    "keeping every claim 100% truthful and grounded in real experience. "
                    "Never fabricate achievements. Only reframe and emphasize what exists."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"""Tailor this resume to the job description below.

JOB DESCRIPTION:
{job_description}

CURRENT RESUME (LaTeX source):
{resume_tex}

Instructions:
1. Identify the 4–6 highest-impact changes: bullet rewrites, keyword insertions,
   Professional Summary edits, or reordering. Prioritize ATS keyword matching.
2. For each change show clearly:
   ORIGINAL: <exact current text>
   REWRITTEN: <new text>
   WHY: one sentence explaining the improvement
3. After the changes, output the COMPLETE updated LaTeX source with all changes
   applied — do not truncate or summarize it.

Format your response exactly as:

## Changes ({{}})

### 1. [Section name] — [one-line reason]
ORIGINAL: ...
REWRITTEN: ...
WHY: ...

(repeat for each change)

## Updated Resume.tex
```latex
[complete updated LaTeX source]
```""",
            }
        ],
    )

    full_text = ""
    for block in response.content:
        if block.type == "text":
            full_text = block.text

    # Print the changes summary; extract and save the LaTeX block
    if "```latex" in full_text:
        latex_open  = full_text.index("```latex")
        summary     = full_text[:latex_open]
        print(summary)

        latex_start = latex_open + len("```latex")
        # Closing fence is optional — use end-of-string if missing (truncated response)
        try:
            latex_end = full_text.index("```", latex_start)
            updated_tex = full_text[latex_start:latex_end].strip()
        except ValueError:
            updated_tex = full_text[latex_start:].strip()
            print("\nWARNING: Response was cut off - LaTeX may be incomplete. "
                  "Try running tailor again if the file looks truncated.\n")

        out_path = os.path.join(script_dir, "Resume_tailored.tex")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(updated_tex)
        print(f"Saved -> Resume_tailored.tex  ({len(updated_tex):,} chars)")
    else:
        # No LaTeX fence at all — print everything
        print(full_text)

    _print_cache_stats(response.usage)


# ── URL Fetcher ───────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Walks HTML and collects visible text, skipping scripts/styles."""
    SKIP = {"script", "style", "head", "noscript", "meta", "link"}

    def __init__(self):
        super().__init__()
        self._depth = 0          # nesting depth inside a skip-tag
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._depth += 1
        if tag in ("p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace while keeping paragraph breaks
        lines = [" ".join(ln.split()) for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]            # drop blank lines
        return "\n".join(lines)


def _fetch_raw(url: str, timeout: int = 15) -> str:
    """
    Fetch a URL and return raw HTML/text. Raises RuntimeError on HTTP/network failure.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def fetch_url(url: str) -> str:
    """
    Download a job-posting URL and return cleaned plain text.

    Strategy:
      1. Try fetching the URL directly and parse the HTML.
      2. If the page looks JS-rendered (too little text), fall back to
         Jina AI's free reader API (r.jina.ai) which handles JS pages.

    Uses only stdlib — no extra packages required.
    Raises RuntimeError with a human-readable message on failure.
    """
    print(f"\nFetching {url} ...")

    # ── Attempt 1: direct fetch ───────────────────────────────────────────────
    try:
        html = _fetch_raw(url)
        extractor = _TextExtractor()
        extractor.feed(html)
        text = extractor.text()
    except RuntimeError as e:
        text = ""
        print(f"  Direct fetch failed ({e}), trying reader API...")

    # ── Attempt 2: Jina reader API (handles JS-rendered pages) ───────────────
    if len(text) < 300:
        if len(text) > 0:
            print("  Page looks JS-rendered, trying reader API...")
        jina_url = f"https://r.jina.ai/{url}"
        try:
            text = _fetch_raw(jina_url, timeout=30)
            # Jina returns markdown — strip any leading metadata lines
            lines = text.splitlines()
            # Drop lines before the first non-empty content line
            text = "\n".join(ln for ln in lines if ln.strip())
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not fetch page via direct or reader API: {e}\n"
                "  -> Paste the job description text manually instead."
            )

    if len(text) < 300:
        raise RuntimeError(
            "Could not extract enough text from the page.\n"
            "  -> Open the page, copy the job description text, and paste it manually."
        )

    # Cap at ~12 000 chars to keep prompts reasonable
    if len(text) > 12_000:
        text = text[:12_000] + "\n[... page truncated ...]"

    print(f"OK  Got {len(text):,} chars.\n")
    return text


# ── Job Search ───────────────────────────────────────────────────────────────

def _fetch_adzuna_jobs(keywords: str, location: str, count: int) -> list[dict]:
    """Call Adzuna API and return a list of normalised job dicts."""
    app_id  = os.getenv("ADZUNA_APP_ID", "")
    api_key = os.getenv("ADZUNA_API_KEY", "")

    if not app_id or not api_key or app_id.startswith("your_"):
        raise RuntimeError("adzuna_not_configured")

    params: dict = {
        "app_id":            app_id,
        "app_key":           api_key,
        "results_per_page":  count,
        "what":              keywords,
        "content-type":      "application/json",
    }
    if location:
        params["where"] = location

    url = (
        "https://api.adzuna.com/v1/api/jobs/us/search/1?"
        + urllib.parse.urlencode(params)
    )

    try:
        raw  = _fetch_raw(url)
        data = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Adzuna API error: {e}")

    jobs = []
    for r in data.get("results", []):
        jobs.append({
            "title":       r.get("title", "Unknown Title"),
            "company":     r.get("company",  {}).get("display_name", "Unknown"),
            "location":    r.get("location", {}).get("display_name", "Unknown"),
            "salary_min":  int(r.get("salary_min") or 0),
            "salary_max":  int(r.get("salary_max") or 0),
            "url":         r.get("redirect_url", ""),
            "description": (r.get("description") or "")[:1500],
        })
    return jobs


def _fetch_remotive_jobs(keywords: str, count: int) -> list[dict]:
    """
    Fallback job source — Remotive free API, no key required.
    Covers remote roles; maps to the same normalised job dict shape.
    """
    # Remotive category that best matches data/analytics/stats keywords
    category = "data"
    params = {"category": category, "limit": count, "search": keywords}
    url = "https://remotive.com/api/remote-jobs?" + urllib.parse.urlencode(params)

    try:
        raw  = _fetch_raw(url)
        data = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Remotive API error: {e}")

    jobs = []
    for r in data.get("jobs", []):
        # Strip HTML tags from description
        extractor = _TextExtractor()
        extractor.feed(r.get("description", ""))
        desc = extractor.text()[:1500]

        salary_str = r.get("salary", "") or ""
        jobs.append({
            "title":       r.get("title", "Unknown Title"),
            "company":     r.get("company_name", "Unknown"),
            "location":    r.get("candidate_required_location", "Remote") or "Remote",
            "salary_min":  0,
            "salary_max":  0,
            "salary_text": salary_str,   # Remotive gives a string, not numbers
            "url":         r.get("url", ""),
            "description": desc,
        })
    return jobs


def _batch_score_jobs(client: anthropic.Anthropic, jobs: list[dict]) -> list[dict]:
    """
    Score all jobs against the resume in one Claude call.
    Returns the same list with a 'score' key added to each item.
    """
    blocks = []
    for i, j in enumerate(jobs, 1):
        blocks.append(
            f"Job {i}: {j['title']} at {j['company']} ({j['location']})\n"
            f"{j['description']}"
        )
    jobs_text = "\n\n---\n\n".join(blocks)

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_base_system(
            "You are a recruiter scoring job listings for a candidate. "
            "Score each 0-100 on how well the candidate qualifies. "
            "60+ = genuinely qualified. 80+ = strong match. Be strict."
        ),
        messages=[{
            "role": "user",
            "content": (
                "Score each job against the resume. "
                "Return ONLY a JSON array — no explanation, no markdown fences.\n"
                'Example: [{"i":1,"score":85},{"i":2,"score":42}]\n\n'
                + jobs_text
            ),
        }],
    )

    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text.strip()

    # Strip any accidental markdown fences
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1].lstrip("json").strip()

    try:
        score_map = {item["i"]: item["score"] for item in json.loads(raw_text)}
    except Exception:
        score_map = {}

    for i, job in enumerate(jobs, 1):
        job["score"] = score_map.get(i, 0)

    _print_cache_stats(response.usage)
    return jobs


def _fmt_salary(job: dict) -> str:
    # Remotive provides a raw string; Adzuna provides numeric min/max
    if job.get("salary_text"):
        return job["salary_text"]
    lo, hi = job.get("salary_min", 0), job.get("salary_max", 0)
    if lo and hi:
        return f"${lo:,.0f} - ${hi:,.0f}"
    if lo:
        return f"${lo:,.0f}+"
    if hi:
        return f"up to ${hi:,.0f}"
    return "not listed"


def search_jobs(client: anthropic.Anthropic) -> None:
    """Prompt for search terms, fetch jobs, batch-score, display ranked matches."""
    print("\n-- Job Search --")
    keywords = input("Keywords (e.g. 'biostatistician', 'health data scientist'): ").strip()
    if not keywords:
        print("No keywords entered.")
        return

    location = input("Location (e.g. 'San Francisco', 'California') or Enter for all: ").strip()

    # Adzuna only accepts geographic locations — "remote" is a keyword, not a place
    REMOTE_WORDS = {"remote", "wfh", "work from home", "work-from-home", "anywhere"}
    if location.lower() in REMOTE_WORDS:
        keywords = keywords + " remote"
        location = ""
        print("(Tip: 'remote' added to keywords — Adzuna uses city/state for location)")

    min_str = input("Minimum fit score to show [default: 60]: ").strip()
    min_score = int(min_str) if min_str.isdigit() else 60

    # Try Adzuna first; fall back to Remotive if keys are missing/invalid
    jobs = []
    source = ""
    try:
        print(f"\nSearching Adzuna for '{keywords}'"
              + (f" in '{location}'" if location else "") + "...")
        jobs = _fetch_adzuna_jobs(keywords, location, count=25)
        source = "Adzuna"
    except RuntimeError as e:
        if "adzuna_not_configured" in str(e):
            print("Adzuna keys not set -- falling back to Remotive (remote jobs only).")
            print("Add ADZUNA_APP_ID + ADZUNA_API_KEY to .env for broader search.")
        else:
            print(f"Adzuna error: {e} -- falling back to Remotive.")
        try:
            jobs = _fetch_remotive_jobs(keywords, count=25)
            source = "Remotive"
        except RuntimeError as e2:
            print(f"\nERROR: {e2}")
            return

    if not jobs:
        print("No listings found. Try different keywords.")
        return

    print(f"Found {len(jobs)} listings from {source}. Scoring against your resume...")
    jobs = _batch_score_jobs(client, jobs)

    matches = sorted(
        [j for j in jobs if j["score"] >= min_score],
        key=lambda x: x["score"],
        reverse=True,
    )

    if not matches:
        print(f"\nNo jobs scored {min_score}+. Try broader keywords or lower the threshold.")
        return

    print(f"\n{len(matches)} match(es) with score >= {min_score}, best first:\n")
    print("=" * 70)
    for i, job in enumerate(matches, 1):
        print(f"\n  {i}.  [{job['score']}/100]  {job['title']}")
        print(f"       Company  : {job['company']}")
        print(f"       Location : {job['location']}")
        print(f"       Salary   : {_fmt_salary(job)}")
        print(f"       URL      : {job['url']}")
    print("\n" + "=" * 70)

    # Let the user drill into any result
    print("\nEnter a job number to score / draft / tailor it, or press Enter to go back:")
    pick = input("Job #: ").strip()
    if not pick.isdigit():
        return

    idx = int(pick) - 1
    if not (0 <= idx < len(matches)):
        print("Invalid number.")
        return

    job = matches[idx]
    print(f"\nFetching full description for: {job['title']} at {job['company']}...")
    try:
        full_jd = fetch_url(job["url"])
    except RuntimeError:
        full_jd = job["description"]   # fall back to Adzuna snippet

    print("\nWhat would you like to do with this job?")
    print("  1  Deep score + gap analysis")
    print("  2  Draft cover letter")
    print("  3  Tailor resume")
    print("  4  All three")
    action = input("\nChoice: ").strip()
    if action in ("1", "4"):
        score_job(client, full_jd)
    if action in ("2", "4"):
        draft_cover_letter(client, full_jd)
    if action in ("3", "4"):
        tailor_resume(client, full_jd)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_cache_stats(usage) -> None:
    created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read    = getattr(usage, "cache_read_input_tokens", 0) or 0
    if created or read:
        print(f"\n[Cache — written: {created:,} tok | read from cache: {read:,} tok]")


def get_job_description() -> str:
    print("\nPaste a job URL  —or—  paste the description text (type END to finish):\n")
    try:
        first = input().strip()
    except EOFError:
        return ""

    # ── URL mode ──────────────────────────────────────────────────────────────
    if first.lower().startswith(("http://", "https://")):
        try:
            return fetch_url(first)
        except RuntimeError as e:
            print(f"\nWARNING: Could not fetch URL: {e}")
            print("\nFall back: paste the job description text, type END when done:\n")
            first = ""   # drop the URL, collect text manually below

    # ── Text mode ─────────────────────────────────────────────────────────────
    lines = [first] if first and first.upper() != "END" else []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Create a .env file in this folder with:\n"
            "    ANTHROPIC_API_KEY=sk-ant-..."
        )
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    client = get_client()

    print("=" * 60)
    print("   JOB AI AGENT  —  Scorer · Drafter · Resume Tailor")
    print("=" * 60)

    while True:
        print("\nWhat would you like to do?")
        print("  1  Score a job (fit score + gap analysis)")
        print("  2  Draft a cover letter")
        print("  3  Tailor my resume to this job")
        print("  4  Score + draft + tailor  (do everything)")
        print("  5  Find matching jobs  (search + auto-score)")
        print("  q  Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            print("Good luck out there!")
            break
        elif choice in ("1", "2", "3", "4"):
            jd = get_job_description()
            if not jd:
                print("Nothing entered -- try again.")
                continue
            if choice in ("1", "4"):
                score_job(client, jd)
            if choice in ("2", "4"):
                draft_cover_letter(client, jd)
            if choice in ("3", "4"):
                tailor_resume(client, jd)
        elif choice == "5":
            search_jobs(client)
        else:
            print("Invalid choice -- enter 1, 2, 3, 4, 5, or q.")


if __name__ == "__main__":
    main()
