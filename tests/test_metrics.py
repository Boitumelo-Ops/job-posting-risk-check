import math

import pytest

from profiler.core import get_nlp, profile
from profiler import lexical as L, rhythm as R, syntax as S


def test_ttr_and_root_ttr():
    toks = "a b a b c".split()
    assert L.ttr(toks) == pytest.approx(3 / 5)
    assert L.root_ttr(toks) == pytest.approx(3 / math.sqrt(5))


def test_yules_k_all_unique_is_zero():
    assert L.yules_k(list("abcdefgh")) == 0


def test_mtld_repetition_is_low():
    assert L.mtld(["the"] * 100) < L.mtld([f"w{i}" for i in range(100)])


def test_mattr_matches_ttr_for_short_text():
    toks = "one two two three".split()
    assert L.mattr(toks, window=50) == L.ttr(toks)


def test_syllables():
    assert L.syllables("cat") == 1
    assert L.syllables("banana") == 3
    assert L.syllables("dedicated") >= 3


def test_burstiness_bounds():
    assert R.burstiness([10, 10, 10]) == -1
    assert -1 <= R.burstiness([1, 40, 2, 35]) <= 1


def test_npvi_constant_is_zero():
    assert R.npvi([2, 2, 2, 2]) == 0


def test_tree_metrics_simple_sentence():
    doc = get_nlp()("The cat sat.")
    s = S.analyse_sentence(next(doc.sents))
    assert s.tree_depth == 2            # sat -> cat -> The
    assert s.n_clauses == 1
    assert s.mean_dep_distance == pytest.approx(1.0)


def test_subordinate_clause_detected():
    doc = get_nlp()("I left because the meeting that she scheduled ran late.")
    s = S.analyse_sentence(next(doc.sents))
    assert s.n_subordinate >= 2         # advcl + relcl


def test_profile_end_to_end():
    p = profile("Short one. This second sentence is quite a bit longer than the first.")
    assert len(p["syntax"]["sentences"]) == 2
    assert p["rhythm"]["sentence_lengths"] == [2, 11]


# ---------------- risk mode ----------------
from profiler.risk import assess, highlight_spans
from profiler.samples import LEGIT_POSTING, SCAM_POSTING


def test_scam_sample_is_high_risk():
    r = assess(SCAM_POSTING)
    assert r["band"] == "High"
    cats = {f["category"] for f in r["flags"]}
    assert {"Upfront payment / fake check", "Requests sensitive personal data",
            "Contact identity mismatch"} <= cats


def test_legit_sample_is_low_risk():
    assert assess(LEGIT_POSTING)["band"] == "Low"


def test_email_not_misread_as_url():
    r = assess("Email jobs.team@gmail.com to apply.")
    assert r["entities"]["urls"] == []
    assert r["entities"]["email_domains"] == ["gmail.com"]


def test_highlight_spans_point_at_evidence():
    r = assess(SCAM_POSTING)
    spans = highlight_spans(SCAM_POSTING, r)
    assert any(SCAM_POSTING[s:e].lower() == "telegram" for s, e, _ in spans)
