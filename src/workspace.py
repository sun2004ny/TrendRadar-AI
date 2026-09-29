"""
workspace.py - Persistent Investigation Desk & Case Management
==============================================================
Manages saved investigations, analyst notes, and case statuses
for continuous monitoring in TrendRadar AI.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

logger = logging.getLogger("TrendRadar.Workspace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_FILE = os.path.join(BASE_DIR, "data", "workspace_investigations.json")


def _load_raw_store() -> Dict[str, Any]:
    if os.path.exists(WORKSPACE_FILE):
        try:
            with open(WORKSPACE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load workspace file: {e}")
    return {"investigations": {}}


def _save_raw_store(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(WORKSPACE_FILE), exist_ok=True)
    with open(WORKSPACE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_investigation(
    trajectory_id: str,
    topic_name: str,
    lifecycle_status: str,
    volume_path: str,
    coherence: float,
    birth_window: str,
    representative_post: str,
    keywords: List[str],
    research_dossier: Optional[Dict[str, Any]] = None,
    ai_brief: Optional[Dict[str, str]] = None,
    analyst_notes: str = "",
    status: str = "Active"
) -> Dict[str, Any]:
    """
    Save or update an investigation in the workspace.
    """
    store = _load_raw_store()
    inv_id = f"INV_{trajectory_id}"
    
    existing = store["investigations"].get(inv_id, {})
    created_at = existing.get("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "id": inv_id,
        "trajectory_id": trajectory_id,
        "topic_name": topic_name,
        "lifecycle_status": lifecycle_status,
        "volume_path": volume_path,
        "avg_coherence": round(float(coherence), 3),
        "birth_window": birth_window,
        "representative_post": representative_post,
        "keywords": keywords,
        "research_dossier": research_dossier or {},
        "ai_brief": ai_brief or {},
        "analyst_notes": analyst_notes or existing.get("analyst_notes", "Initial observation filed."),
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at
    }

    store["investigations"][inv_id] = record
    _save_raw_store(store)
    return record


def list_investigations() -> List[Dict[str, Any]]:
    """Return all saved investigations ordered by updated_at descending."""
    store = _load_raw_store()
    records = list(store.get("investigations", {}).values())
    records.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return records


def get_investigation(inv_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an investigation by ID."""
    store = _load_raw_store()
    return store.get("investigations", {}).get(inv_id)


def update_investigation_status(inv_id: str, status: str, notes: Optional[str] = None) -> bool:
    """Update status or notes for an existing investigation."""
    store = _load_raw_store()
    if inv_id in store["investigations"]:
        store["investigations"][inv_id]["status"] = status
        if notes is not None:
            store["investigations"][inv_id]["analyst_notes"] = notes
        store["investigations"][inv_id]["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        _save_raw_store(store)
        return True
    return False


def delete_investigation(inv_id: str) -> bool:
    """Delete an investigation from the workspace."""
    store = _load_raw_store()
    if inv_id in store["investigations"]:
        del store["investigations"][inv_id]
        _save_raw_store(store)
        return True
    return False


def export_workspace_markdown() -> str:
    """Export all investigations into a consolidated executive dossier."""
    investigations = list_investigations()
    if not investigations:
        return "# TrendRadar AI Workspace Dossier\n\n*No investigations currently logged.*"

    md = "# TrendRadar AI: Master Workspace Dossier\n\n"
    md += f"**Export Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
    md += f"**Total Monitored Cases**: {len(investigations)}\n\n---\n\n"

    for inv in investigations:
        md += f"## [{inv['id']}] {inv['topic_name']} ({inv['status']})\n\n"
        md += f"- **Trajectory ID**: `{inv['trajectory_id']}` | **Lifecycle**: `{inv['lifecycle_status']}`\n"
        md += f"- **Volume History**: `{inv['volume_path']}` | **Coherence**: `{inv['avg_coherence']}`\n"
        md += f"- **Representative Post**: *\"{inv['representative_post']}\"*\n"
        md += f"- **Analyst Notes**: {inv['analyst_notes']}\n\n"

        brief = inv.get("ai_brief", {})
        if brief.get("what_is_happening"):
            md += "### AI Briefing Summary\n"
            md += f"- **What**: {brief.get('what_is_happening')}\n"
            md += f"- **Why**: {brief.get('why_is_it_emerging')}\n"
            md += f"- **Next Steps**: {brief.get('recommended_next_steps')}\n\n"
        md += "---\n\n"

    return md
