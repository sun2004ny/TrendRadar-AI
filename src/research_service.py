"""
research_service.py - External Context & Multi-Source Evidence Retrieval
========================================================================
Pulls real-world corroboration from Wikipedia, arXiv, and public knowledge
APIs for any detected topic from the NLP pipeline.

Designed to be lightweight (using Python standard library urllib/json/xml)
with zero heavy dependencies.
"""

import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Any

logger = logging.getLogger("TrendRadar.ResearchService")
USER_AGENT = "TrendRadarAI/2.0 (Research & Intelligence Engine; mailto:research@trendradar.ai)"


def fetch_wikipedia_context(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search Wikipedia for background context on a detected topic or entity.
    Returns executive summary, article title, and reference URL.
    """
    clean_query = query.strip()
    if not clean_query:
        return {"found": False, "summary": "", "url": "", "title": ""}

    encoded_query = urllib.parse.quote(clean_query)
    search_url = (
        f"https://en.wikipedia.org/w/api.php?"
        f"action=query&list=search&srsearch={encoded_query}&utf8=1&format=json&srlimit={max_results}"
    )

    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return {"found": False, "summary": "No direct Wikipedia article found.", "url": "", "title": clean_query}

        top_hit = search_results[0]
        title = top_hit["title"]

        # Fetch clean intro extract
        encoded_title = urllib.parse.quote(title)
        extract_url = (
            f"https://en.wikipedia.org/w/api.php?"
            f"action=query&prop=extracts&exintro=1&explaintext=1&titles={encoded_title}&format=json"
        )
        req_extract = urllib.request.Request(extract_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req_extract, timeout=5) as response:
            extract_data = json.loads(response.read().decode("utf-8"))

        pages = extract_data.get("query", {}).get("pages", {})
        extract_text = ""
        for page_id, page_info in pages.items():
            if page_id != "-1":
                extract_text = page_info.get("extract", "")
                break

        page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
        return {
            "found": bool(extract_text),
            "title": title,
            "summary": extract_text[:600] + ("..." if len(extract_text) > 600 else ""),
            "url": page_url,
            "related_topics": [r["title"] for r in search_results[1:max_results]]
        }

    except Exception as e:
        logger.warning(f"Wikipedia lookup error for '{clean_query}': {e}")
        return {
            "found": False,
            "summary": f"Could not retrieve Wikipedia context ({str(e)[:50]}).",
            "url": "",
            "title": clean_query
        }


def fetch_arxiv_papers(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search arXiv for academic research and empirical preprints matching the topic.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    encoded_query = urllib.parse.quote(clean_query)
    arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"

    papers = []
    try:
        req = urllib.request.Request(arxiv_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=6) as response:
            xml_data = response.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            link = entry.find("atom:id", ns)
            author = entry.find("atom:author/atom:name", ns)

            if title is not None and title.text:
                clean_title = " ".join(title.text.split())
                clean_summary = " ".join(summary.text.split()) if summary is not None and summary.text else ""
                papers.append({
                    "title": clean_title,
                    "summary": clean_summary[:350] + ("..." if len(clean_summary) > 350 else ""),
                    "published": published.text[:10] if published is not None and published.text else "N/A",
                    "author": author.text if author is not None and author.text else "Researcher et al.",
                    "url": link.text.strip() if link is not None and link.text else ""
                })
    except Exception as e:
        logger.warning(f"arXiv lookup error for '{clean_query}': {e}")

    return papers


def get_topic_research_dossier(topic_name: str, keywords: List[str], representative_post: str) -> Dict[str, Any]:
    """
    Comprehensive research aggregator.
    Builds a unified evidence package combining Wikipedia, arXiv, and entity signals.
    """
    # Formulate search queries
    primary_query = topic_name.replace("/", " ").replace("_", " ")
    
    # Strip common non-topical terms
    stop_words = {"discussion", "topic", "post", "question", "people", "would", "think", "what", "could"}
    kw_filtered = [k for k in keywords if k.lower() not in stop_words]
    search_query = " ".join(kw_filtered[:3]) if kw_filtered else primary_query

    wiki_res = fetch_wikipedia_context(search_query)
    if not wiki_res.get("found") and primary_query != search_query:
        wiki_res = fetch_wikipedia_context(primary_query)

    arxiv_res = fetch_arxiv_papers(search_query, max_results=2)

    # Corroboration analysis
    is_corroborated = wiki_res.get("found", False) or len(arxiv_res) > 0
    confidence_level = "High External Corroboration" if (wiki_res.get("found") and len(arxiv_res) > 0) else (
        "Verified External Topic" if wiki_res.get("found") else "Platform-Specific Discourse"
    )

    return {
        "search_query": search_query,
        "wikipedia": wiki_res,
        "arxiv_papers": arxiv_res,
        "corroboration_confidence": confidence_level,
        "has_external_evidence": is_corroborated
    }
