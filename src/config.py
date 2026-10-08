"""
src/config.py - Centralized configuration loader for TrendRadar-AI
Loads settings from config.yaml using pathlib with typed access and fallbacks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# Repository root directory using pathlib
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


class ConfigNamespace:
    """Convenience wrapper providing attribute access to nested dictionaries."""
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)
        self._data = data

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data


_CACHED_CONFIG: Optional[ConfigNamespace] = None


def load_raw_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Reads YAML file directly with robust fallback defaults."""
    target_path = path or CONFIG_PATH
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.warning("Could not read %s: %s. Using default config.", target_path, e)

    # Defaults if file is missing
    return {
        "dataset": {
            "name": "SocialGrep/one-million-reddit-questions",
            "time_resolution": "daily",
            "total_checkpoints": 30,
        },
        "clustering": {
            "svd_components": 12,
            "min_cluster_size": 7,
            "min_samples": 3,
            "cluster_selection_method": "leaf",
            "metric": "euclidean",
            "spam_ratio_threshold": 0.35,
            "max_cluster_size": 150,
            "coherence_min": 0.45,
            "c_tf_idf_top_k": 4,
        },
        "temporal": {
            "match_threshold": 0.68,
            "keyword_jaccard_threshold": 0.20,
            "entity_jaccard_threshold": 0.20,
            "centroid_ema_alpha": 0.65,
            "max_gap_checkpoints": 1,
        },
        "emergence": {
            "coherence_floor": 0.62,
            "min_coherence_floor": 0.60,
            "burst_topic_share": 0.020,
            "burst_static_volume": 15,
            "growth_ratio_threshold": 1.3,
            "seed_promotion_coherence": 0.85,
            "quarantine_volume_min": 3,
            "quarantine_volume_max": 6,
        },
        "rolling_buffer": {
            "enabled": False,
            "window_hours": 72,
            "min_sparse_posts": 1,
            "incubation_posts_threshold": 2,
            "confirmed_seed_threshold": 3,
            "semantic_similarity_threshold": 0.68,
            "time_decay_factor": 0.85,
            "coherence_floor": 0.65,
        },
        "baseline": {
            "min_freq": 5,
            "surge_threshold": 2.0,
            "z_score_threshold": 3.0,
            "ngram_range": [1, 2],
        },
        "evaluation": {
            "tuning_days": [f"D_{i:02d}" for i in range(1, 21)],
            "held_out_days": [f"D_{i:02d}" for i in range(21, 31)],
            "lead_window_days_prior": 2,
            "lag_window_days_post": 3,
            "min_keyword_matches": 1,
        },
    }


def get_config(reload: bool = False, config_path: Optional[Path] = None) -> ConfigNamespace:
    """Returns singleton ConfigNamespace instance."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or reload or config_path is not None:
        raw = load_raw_config(config_path)
        _CACHED_CONFIG = ConfigNamespace(raw)
    return _CACHED_CONFIG
