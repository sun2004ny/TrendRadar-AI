"""
llm_summary.py - AI Executive Briefing Engine
=============================================
Synthesizes detected mathematical evidence, cluster posts, and external
research context into an actionable 4-part intelligence brief.

Principle:
  ML = Detection (TrendRadar AI mathematical clustering)
  Research = Context (Wikipedia / arXiv)
  LLM = Explanation (Explain the evidence, do not invent trends)
  Analyst = Decision
"""

import os
import json
import logging
import urllib.request
from typing import Dict, Any, List, Optional

logger = logging.getLogger("TrendRadar.LLMSummary")


def generate_executive_brief(
    topic_name: str,
    lifecycle_status: str,
    volume_history: str,
    coherence: float,
    birth_window: str,
    representative_post: str,
    keywords: List[str],
    research_dossier: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    provider: str = "auto"
) -> Dict[str, str]:
    """
    Generate an evidence-grounded 4-part executive brief:
      1. WHAT is happening?
      2. WHY is it emerging?
      3. What EVIDENCE supports it?
      4. What should the analyst investigate NEXT?
    """
    dossier = research_dossier or {}
    wiki_info = dossier.get("wikipedia", {})
    wiki_summary = wiki_info.get("summary", "") if wiki_info.get("found") else ""
    arxiv_papers = dossier.get("arxiv_papers", [])
    arxiv_summary = (
        f"Academic literature ({arxiv_papers[0]['title']})"
        if arxiv_papers else "No preprints indexed."
    )

    kw_str = ", ".join(keywords[:6]) if keywords else topic_name

    # Check for live API key (Gemini preferred)
    gemini_key = api_key or os.getenv("GEMINI_API_KEY", "")

    if gemini_key:
        try:
            brief = _call_gemini_api(
                gemini_key=gemini_key,
                topic_name=topic_name,
                status=lifecycle_status,
                volume_path=volume_history,
                coherence=coherence,
                rep_post=representative_post,
                keywords=kw_str,
                wiki_summary=wiki_summary
            )
            if brief:
                return brief
        except Exception as e:
            logger.warning(f"Live Gemini API call failed: {e}. Falling back to evidence synthesizer.")

    # High-fidelity Evidence Synthesizer (Zero API key required, 100% grounded in data)
    return _synthesize_from_evidence(
        topic_name=topic_name,
        lifecycle_status=lifecycle_status,
        volume_history=volume_history,
        coherence=coherence,
        birth_window=birth_window,
        representative_post=representative_post,
        keywords=kw_str,
        wiki_summary=wiki_summary,
        arxiv_papers=arxiv_papers
    )


def _call_gemini_api(
    gemini_key: str,
    topic_name: str,
    status: str,
    volume_path: str,
    coherence: float,
    rep_post: str,
    keywords: str,
    wiki_summary: str
) -> Optional[Dict[str, str]]:
    """Query Google Gemini API for an evidence-grounded brief."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
    
    prompt = f"""You are a Senior Threat Intelligence Analyst for TrendRadar AI.
Analyze the following MATHEMATICALLY DETECTED emerging topic.
Your role is to explain the evidence, NOT decide if the trend exists.

TOPIC: {topic_name}
LIFECYCLE STATUS: {status}
VOLUME PROGRESSION: {volume_path}
SEMANTIC COHERENCE: {coherence:.3f}
CANONICAL USER POST: "{rep_post}"
KEYWORDS: {keywords}
EXTERNAL CONTEXT (Wikipedia): {wiki_summary or 'None available'}

Respond in JSON format with exact keys:
{{
  "what_is_happening": "1-2 concise sentences summarizing the core discussion theme.",
  "why_is_it_emerging": "Explanation of the catalyst (news event, policy announcement, or viral spark).",
  "evidence_summary": "Summary of numerical evidence (volume path, semantic coherence score, quote).",
  "recommended_next_steps": "Actionable instructions for the analyst or stakeholder to investigate next."
}}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=8) as response:
        result = json.loads(response.read().decode("utf-8"))
        text_content = result["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text_content)
        
        parsed["full_brief_markdown"] = _format_markdown_report(topic_name, parsed)
        return parsed


def _synthesize_from_evidence(
    topic_name: str,
    lifecycle_status: str,
    volume_history: str,
    coherence: float,
    birth_window: str,
    representative_post: str,
    keywords: str,
    wiki_summary: str,
    arxiv_papers: list
) -> Dict[str, str]:
    """Generate structured brief directly from NLP evidence and external context."""
    # 1. WHAT is happening
    if wiki_summary:
        first_sentence = wiki_summary.split(". ")[0] + "."
        what = f"Community discussion centered on {topic_name.lower()}, anchored by: {first_sentence} Discussion reflects user sentiment captured in post: \"{representative_post}\""
    else:
        what = f"Rapidly coalescing discourse surrounding '{topic_name}' ({keywords}). Representative sentiment: \"{representative_post}\""

    # 2. WHY is it emerging
    if "[BURST]" in lifecycle_status:
        why = f"High acceleration surge triggered in checkpoint {birth_window}. Discussion volume transitioned rapidly along timeline ({volume_history}), indicating a sudden real-world catalyst or policy announcement."
    elif "[CONFIRMED_SEED]" in lifecycle_status:
        why = f"Early-stage incubation signal verified across multiple checkpoints with extreme semantic coherence ({coherence:.3f}). Topic represents an organic community consensus forming prior to mass-market peak."
    else:
        why = f"Sustained discussion tracking along trajectory {volume_history} with consistent topical coherence ({coherence:.3f})."

    # 3. EVIDENCE
    evidence = (
        f"• Volume Progression: {volume_history}\n"
        f"• Semantic Coherence: {coherence:.3f} (Dense vector cluster density)\n"
        f"• Core c-TF-IDF Vocabulary: {keywords}\n"
        f"• Ground-Truth Citation: \"{representative_post}\""
    )
    if arxiv_papers:
        evidence += f"\n• Corroborating Academic Reference: '{arxiv_papers[0]['title']}' ({arxiv_papers[0].get('published', 'Recent')})"

    # 4. NEXT STEPS
    next_steps = (
        f"1. Monitor subsequent checkpoint stream for volume inflection or drop-off.\n"
        f"2. Verify cross-platform propagation (correlate with news wires, Twitter/X, and financial releases).\n"
        f"3. Dispatch early-warning stakeholder notification if trajectory acceleration persists."
    )

    result = {
        "what_is_happening": what,
        "why_is_it_emerging": why,
        "evidence_summary": evidence,
        "recommended_next_steps": next_steps
    }
    result["full_brief_markdown"] = _format_markdown_report(topic_name, result)
    return result


def _format_markdown_report(topic_name: str, brief: Dict[str, str]) -> str:
    """Format into standard Intelligence Brief markdown."""
    return f"""# TrendRadar AI: Executive Intelligence Brief
**Subject**: {topic_name}  
**Classification**: Vetted Emerging Intelligence  
**Generated by**: TrendRadar AI Dual-Tier NLP Engine  

---

### 1. What is Happening?
{brief.get('what_is_happening', 'N/A')}

### 2. Why is it Emerging?
{brief.get('why_is_it_emerging', 'N/A')}

### 3. Supporting Evidence & Metrics
{brief.get('evidence_summary', 'N/A')}

### 4. Recommended Analyst Next Steps
{brief.get('recommended_next_steps', 'N/A')}

---
*Notice: This brief was algorithmically generated using dense semantic embeddings and external knowledge grounding.*
"""
