"""Single entry point: profile(text) -> dict with syntax, lexical and rhythm layers."""
from __future__ import annotations

from functools import lru_cache

import spacy

from .lexical import document_lexical
from .rhythm import document_rhythm
from .syntax import document_syntax


@lru_cache(maxsize=1)
def get_nlp(model: str = "en_core_web_sm"):
    return spacy.load(model)


def profile(text: str, model: str = "en_core_web_sm") -> dict:
    doc = get_nlp(model)(text)
    return {
        "syntax": document_syntax(doc),
        "lexical": document_lexical(doc),
        "rhythm": document_rhythm(doc),
    }


# Metrics exposed on the comparison radar, with a plausible range for normalisation.
RADAR_METRICS = {
    "Sentence length": ("syntax", "mean_sentence_length", 5, 40),
    "Tree depth": ("syntax", "mean_tree_depth", 2, 10),
    "Dependency distance": ("syntax", "mean_dep_distance", 1.5, 4.5),
    "Subordination": ("syntax", "subordination_index", 0, 0.6),
    "Lexical diversity (MTLD)": ("lexical", "mtld", 20, 140),
    "Lexical density": ("lexical", "lexical_density", 0.35, 0.65),
    "Rhythmic variation (CV)": ("rhythm", "sentence_length_cv", 0, 1.2),
    "Stress alternation": ("rhythm", "stress_alternation", 0.3, 0.8),
}


def radar_vector(p: dict) -> dict:
    out = {}
    for label, (layer, key, lo, hi) in RADAR_METRICS.items():
        v = p[layer]["summary"].get(key) or 0.0
        out[label] = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return out
