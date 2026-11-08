"""
Lexical complexity & diversity.

Notation: N = number of tokens, V = number of types (distinct words),
V(m, N) = number of types occurring exactly m times (the frequency spectrum).
Plain TTR = V/N falls as N grows (Heaps' law: V ~ K * N^beta, beta < 1), so most
of the measures below exist to correct for text length.
"""
from __future__ import annotations

import math
import re
from collections import Counter

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


# ---------- syllables (heuristic; good to ~90% on English) ----------
_VOWELS = re.compile(r"[aeiouy]+")


def syllables(word: str) -> int:
    w = word.lower().strip(".,;:!?\"'()")
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$", "", w)  # silent endings
    w = re.sub(r"^y", "", w)
    return max(1, len(_VOWELS.findall(w)))


# ---------- diversity ----------
def ttr(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def root_ttr(tokens):
    """Guiraud's R = V / sqrt(N)."""
    return len(set(tokens)) / math.sqrt(len(tokens)) if tokens else 0.0


def mattr(tokens, window=50):
    """Moving-average TTR (Covington & McFall 2010): mean TTR over every window of size w."""
    n = len(tokens)
    if n < window:
        return ttr(tokens)
    counts = Counter(tokens[:window])
    total = len(counts) / window
    for i in range(window, n):
        out, inc = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] += 1
        total += len(counts) / window
    return total / (n - window + 1)


def _mtld_pass(tokens, threshold):
    factors, types, count = 0.0, set(), 0
    for tok in tokens:
        types.add(tok)
        count += 1
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count:  # partial factor for the remainder
        rem_ttr = len(types) / count
        factors += (1 - rem_ttr) / (1 - threshold)
    return len(tokens) / factors if factors else float(len(tokens))


def mtld(tokens, threshold=0.72):
    """
    MTLD (McCarthy & Jarvis 2010): mean length of sequential runs that keep TTR above 0.72.
    Computed forwards and backwards, then averaged. Robust to text length.
    """
    if not tokens:
        return 0.0
    return (_mtld_pass(tokens, threshold) + _mtld_pass(tokens[::-1], threshold)) / 2


def hdd(tokens, sample=42):
    """
    HD-D (McCarthy & Jarvis 2007): expected TTR of a random 42-token sample, via the
    hypergeometric distribution. For each type with frequency f, P(appears in sample)
    = 1 - C(N-f, s) / C(N, s); contribution = P / s.
    """
    n = len(tokens)
    if n < sample:
        return ttr(tokens)
    denom = math.comb(n, sample)
    return sum((1 - math.comb(n - f, sample) / denom) / sample for f in Counter(tokens).values())


def yules_k(tokens):
    """
    Yule's K = 10^4 * (sum_m m^2 V(m,N) - N) / N^2.
    Measures repetitiveness; largely independent of N. Higher K = less diverse.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    spectrum = Counter(Counter(tokens).values())
    s2 = sum(m * m * v for m, v in spectrum.items())
    return 1e4 * (s2 - n) / (n * n)


def honore_r(tokens):
    """Honoré's R = 100 * log N / (1 - V1/V). Rewards hapax legomena (words used once)."""
    n, freqs = len(tokens), Counter(tokens)
    v, v1 = len(freqs), sum(1 for c in freqs.values() if c == 1)
    if n == 0 or v1 == v:
        return float("inf") if n else 0.0
    return 100 * math.log(n) / (1 - v1 / v)


# ---------- readability & density ----------
def lexical_density(doc):
    words = [t for t in doc if t.is_alpha]
    return sum(t.pos_ in CONTENT_POS for t in words) / len(words) if words else 0.0


def readability(words, n_sentences):
    if not words or not n_sentences:
        return {"flesch_reading_ease": 0.0, "fk_grade": 0.0}
    syl = sum(syllables(w) for w in words)
    wps, spw = len(words) / n_sentences, syl / len(words)
    return {
        "flesch_reading_ease": 206.835 - 1.015 * wps - 84.6 * spw,
        "fk_grade": 0.39 * wps + 11.8 * spw - 15.59,
    }


def document_lexical(doc) -> dict:
    words = [t.text for t in doc if t.is_alpha]
    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
    n_sents = sum(1 for _ in doc.sents)
    freqs = Counter(lemmas)
    hr = honore_r(lemmas)
    return {
        "summary": {
            "tokens": len(lemmas),
            "types": len(freqs),
            "ttr": ttr(lemmas),
            "root_ttr": root_ttr(lemmas),
            "mattr": mattr(lemmas),
            "mtld": mtld(lemmas),
            "hdd": hdd(lemmas),
            "yules_k": yules_k(lemmas),
            "honore_r": hr if math.isfinite(hr) else None,
            "lexical_density": lexical_density(doc),
            "mean_word_length": sum(map(len, words)) / len(words) if words else 0.0,
            **readability(words, n_sents),
        },
        "frequencies": freqs.most_common(),
    }
