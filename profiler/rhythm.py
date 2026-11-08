"""
Prosodic & rhythmic profile of prose.

Prose rhythm lives at two scales:
  * macro  - how sentence lengths vary across the text (pacing / cadence)
  * micro  - how stressed and unstressed syllables alternate inside a sentence

We have no audio, so stress is approximated: content words carry a primary stress
on their first syllable, function words are unstressed. Crude, but it reproduces
the main contrast between 'heavy' noun-dense prose and 'light' function-word prose.
Swap in the CMU Pronouncing Dictionary (`pip install pronouncing`) for real lexical stress.
"""
from __future__ import annotations

import math
import re
from statistics import mean, pstdev

from .lexical import CONTENT_POS, syllables


def coefficient_of_variation(xs):
    m = mean(xs)
    return pstdev(xs) / m if m else 0.0


def burstiness(xs):
    """Goh & Barabási (2008): B = (sigma - mu) / (sigma + mu). -1 = metronomic, 0 = random, +1 = bursty."""
    m, s = mean(xs), pstdev(xs)
    return (s - m) / (s + m) if (s + m) else 0.0


def autocorrelation(xs, lag=1):
    """Lag-k autocorrelation: do long sentences cluster (>0) or alternate with short ones (<0)?"""
    n = len(xs)
    if n <= lag + 1:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs)
    if var == 0:
        return 0.0
    return sum((xs[i] - m) * (xs[i + lag] - m) for i in range(n - lag)) / var


def npvi(xs):
    """
    Normalised Pairwise Variability Index (Grabe & Low 2002):
    nPVI = 100/(n-1) * sum |d_k - d_{k+1}| / ((d_k + d_{k+1})/2)
    Classically computed on vowel durations; here on syllables-per-word as a proxy.
    High = 'stress-timed' contrastive rhythm, low = even 'syllable-timed' flow.
    """
    pairs = [(a, b) for a, b in zip(xs, xs[1:]) if a + b]
    if not pairs:
        return 0.0
    return 100 * mean(abs(a - b) / ((a + b) / 2) for a, b in pairs)


def stress_pattern(sent) -> list[int]:
    """Binary stress string, one entry per syllable (1 = stressed)."""
    pattern = []
    for t in sent:
        if not t.is_alpha:
            continue
        n = syllables(t.text)
        if t.pos_ in CONTENT_POS:
            pattern += [1] + [0] * (n - 1)
        else:
            pattern += [0] * n
    return pattern


def alternation_rate(pattern):
    """Share of adjacent syllable pairs that switch stress. Perfect iambs/trochees = 1.0."""
    if len(pattern) < 2:
        return 0.0
    return sum(a != b for a, b in zip(pattern, pattern[1:])) / (len(pattern) - 1)


def entropy(xs):
    """Shannon entropy (bits) of a discrete distribution - here, of sentence lengths."""
    from collections import Counter
    c, n = Counter(xs), len(xs)
    return -sum((v / n) * math.log2(v / n) for v in c.values()) if n else 0.0


_PHRASE_SPLIT = re.compile(r"[,;:—–()]|\s-\s")


def document_rhythm(doc) -> dict:
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    lengths = [sum(t.is_alpha for t in s) for s in sents]
    patterns = [stress_pattern(s) for s in sents]
    word_syl = [syllables(t.text) for t in doc if t.is_alpha]
    phrases = [len(p.split()) for s in sents for p in _PHRASE_SPLIT.split(s.text) if p.strip()]
    if not lengths:
        return {"summary": {}, "sentence_lengths": [], "stress": []}
    return {
        "summary": {
            "sentence_length_cv": coefficient_of_variation(lengths),
            "burstiness": burstiness(lengths),
            "lag1_autocorrelation": autocorrelation(lengths),
            "length_entropy_bits": entropy(lengths),
            "npvi": npvi(word_syl),
            "stress_alternation": mean(alternation_rate(p) for p in patterns),
            "stress_density": sum(map(sum, patterns)) / max(1, sum(map(len, patterns))),
            "mean_phrase_length": mean(phrases) if phrases else 0.0,
        },
        "sentence_lengths": lengths,
        "stress": patterns,
        "phrase_lengths": phrases,
    }
