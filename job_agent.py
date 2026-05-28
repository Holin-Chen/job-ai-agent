#!/usr/bin/env python3
"""
Job AI Agent — Scorer + Cover Letter Drafter
─────────────────────────────────────────────
Usage:
    python job_agent.py

Requires a .env file (or exported env var) with:
    ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

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
    print("\n⏳  Analyzing fit (may take 20–40 s with adaptive thinking)…\n")

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
    print("\n✍️  Drafting cover letter…\n")
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_cache_stats(usage) -> None:
    created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read    = getattr(usage, "cache_read_input_tokens", 0) or 0
    if created or read:
        print(f"\n[Cache — written: {created:,} tok | read from cache: {read:,} tok]")


def get_job_description() -> str:
    print("\nPaste the job description below.")
    print("Type END on its own line when finished:\n")
    lines = []
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
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
    print("   JOB AI AGENT  —  Scorer + Cover Letter Drafter")
    print("=" * 60)

    while True:
        print("\nWhat would you like to do?")
        print("  1  Score a job (fit score + gap analysis)")
        print("  2  Draft a cover letter")
        print("  3  Score AND draft")
        print("  q  Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            print("Good luck out there!")
            break
        elif choice in ("1", "2", "3"):
            jd = get_job_description()
            if not jd:
                print("Nothing entered — try again.")
                continue
            if choice in ("1", "3"):
                score_job(client, jd)
            if choice in ("2", "3"):
                draft_cover_letter(client, jd)
        else:
            print("Invalid choice — enter 1, 2, 3, or q.")


if __name__ == "__main__":
    main()
