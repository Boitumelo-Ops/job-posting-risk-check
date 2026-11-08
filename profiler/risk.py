"""
Job-posting scam risk analysis.

Maps onto the NLP pipeline from Trailhead's "Natural Language Parsing" unit:
  * Named Entity Recognition   -> who/what/how-much the posting claims (ORG, MONEY, contacts)
  * Intent analysis            -> rule-based matchers over lemmas (urgency, payment, PII, off-platform)
  * Sentiment analysis         -> VADER polarity + hype markers (exclamations, caps)
  * Pragmatics / context       -> what a legitimate posting normally contains, and what is missing
  * Syntax / style             -> template-like uniformity from the stylometric profiler

The score is a transparent, weighted rule system - every point is tied to evidence in the text.
It is a triage aid, not a verdict: treat it as "how carefully should I check this?"
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from spacy.matcher import PhraseMatcher
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .core import get_nlp
from .lexical import mtld
from .rhythm import burstiness

# ------------------------------------------------------------------ patterns
FREE_EMAIL = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "icloud.com",
    "proton.me", "protonmail.com", "gmx.com", "mail.com", "yandex.com", "live.com",
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "shorturl.at"}

EMAIL_RE = re.compile(r"\b[\w.+-]+@([\w-]+\.)+[a-z]{2,}\b", re.I)
URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,})(/\S*)?", re.I)
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
PAY_RE = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*(?:-|to)?\s*\$?\s?(\d{1,3}(?:,\d{3})*|\d+)?(?:\.\d+)?\s*"
    r"(?:/|per|an|a)?\s*(hour|hr|week|wk|day|month|year|yr)?",
    re.I,
)

# Intent lexicon: category -> (weight per hit, cap, phrases). Matched case-insensitively on the text.
INTENTS = {
    "Upfront payment / fake check": (18, 36, [
        "send you a check", "deposit the check", "mobile deposit", "purchase equipment",
        "buy equipment", "approved vendor", "reimbursed", "will be reimbursed", "gift card",
        "gift cards", "wire transfer", "western union", "zelle", "cash app", "bitcoin",
        "crypto", "training fee", "registration fee", "starter kit", "pay for your",
    ]),
    "Requests sensitive personal data": (15, 30, [
        "social security", "ssn", "bank account", "account number", "routing number",
        "driver's license", "drivers license", "copy of your id", "photo of your id",
        "date of birth", "credit card", "online banking", "login details",
    ]),
    "Off-platform / chat-only contact": (12, 24, [
        "telegram", "whatsapp", "signal app", "google hangouts", "hangouts", "wickr",
        "text me", "text us", "chat interview", "interview via chat", "instant message",
        "skype id", "teams chat only",
    ]),
    "Urgency / pressure": (7, 21, [
        "immediately", "urgent", "urgently", "asap", "act now", "limited slots",
        "limited positions", "within 24 hours", "today only", "don't miss", "hurry",
        "first come first serve", "first come, first served", "start today", "start immediately",
    ]),
    "Too good to be true": (8, 24, [
        "no experience", "no experience needed", "no experience required", "no interview",
        "hired on the spot", "guaranteed income", "easy money", "earn up to", "work from home",
        "be your own boss", "flexible hours", "weekly pay", "daily pay", "100% remote",
        "no skills required", "anyone can do",
    ]),
    "Vague role / reshipping": (10, 20, [
        "package forwarding", "reshipping", "receive packages", "reship", "payment processing agent",
        "process payments", "financial agent", "money transfer agent", "personal assistant",
        "data entry clerk",
    ]),
}

# Pragmatic expectations: what a legitimate posting usually contains.
EXPECTED_SECTIONS = {
    "Responsibilities": ["responsibilit", "you will", "duties", "what you'll do", "day-to-day"],
    "Qualifications": ["qualification", "requirement", "experience with", "years of", "degree", "certification"],
    "Company identity": ["about us", "our company", "we are a", "founded", "headquarter", "our team"],
    "Application process": ["apply", "application", "interview process", "careers page", "applicant"],
}


_INTENT_WORDS = {p for _, _, ps in INTENTS.values() for p in ps}


@dataclass
class Flag:
    category: str
    points: float
    detail: str
    evidence: list[str]


# ------------------------------------------------------------------ helpers
_vader = None


def _sentiment(text: str) -> dict:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader.polarity_scores(text)


_matchers: dict[int, PhraseMatcher] = {}


def _intent_matcher(nlp) -> PhraseMatcher:
    key = id(nlp)
    if key not in _matchers:
        m = PhraseMatcher(nlp.vocab, attr="LOWER")
        for cat, (_, _, phrases) in INTENTS.items():
            m.add(cat, [nlp.make_doc(p) for p in phrases])
        _matchers[key] = m
    return _matchers[key]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract_entities(doc) -> dict:
    text = doc.text
    emails = [m.group(0) for m in EMAIL_RE.finditer(text)]
    email_domains = {e.split("@")[1].lower() for e in emails}
    urls = []
    for m in URL_RE.finditer(EMAIL_RE.sub(" ", text)):  # don't read emails as URLs
        dom = m.group(1).lower()
        if dom not in email_domains and not any(e.endswith(dom) for e in emails):
            urls.append(dom)
    return {
        # drop ORG hits that are really chat apps / payment rails from the intent lexicon
        "organizations": sorted({e.text for e in doc.ents if e.label_ == "ORG"
                                 and e.text.lower() not in _INTENT_WORDS}),
        "people": sorted({e.text for e in doc.ents if e.label_ == "PERSON"}),
        "places": sorted({e.text for e in doc.ents if e.label_ in {"GPE", "LOC"}}),
        "money": [e.text for e in doc.ents if e.label_ == "MONEY"],
        "emails": emails,
        "email_domains": sorted(email_domains),
        "urls": sorted(set(urls)),
        "phones": [m.group(0).strip() for m in PHONE_RE.finditer(text)],
    }


def _hourly_estimate(text: str) -> float | None:
    """Rough annualised -> hourly conversion of the highest pay figure, for 'too good' checks."""
    best = None
    for m in PAY_RE.finditer(text):
        hi = m.group(2) or m.group(1)
        try:
            val = float(hi.replace(",", ""))
        except (TypeError, ValueError):
            continue
        unit = (m.group(3) or "").lower()
        per_hour = {"hour": 1, "hr": 1, "day": 8, "week": 40, "wk": 40, "month": 173,
                    "year": 2080, "yr": 2080}.get(unit)
        if per_hour is None:
            continue
        hourly = val / per_hour
        best = hourly if best is None else max(best, hourly)
    return best


# ------------------------------------------------------------------ main
def assess(text: str) -> dict:
    nlp = get_nlp()
    doc = nlp(text)
    low = text.lower()
    flags: list[Flag] = []
    ents = extract_entities(doc)

    # 1. Intent analysis (phrase matcher over lowercase tokens)
    hits: dict[str, list[str]] = {}
    for match_id, start, end in _intent_matcher(nlp)(doc):
        cat = nlp.vocab.strings[match_id]
        hits.setdefault(cat, []).append(doc[start:end].text)
    for cat, found in hits.items():
        weight, cap, _ = INTENTS[cat]
        uniq = sorted(set(f.lower() for f in found))
        flags.append(Flag(cat, min(cap, weight * len(uniq)),
                          f"{len(uniq)} distinct cue(s) found", uniq))

    # 2. NER consistency checks
    free = [d for d in ents["email_domains"] if d in FREE_EMAIL]
    if free:
        claim = f" while naming {', '.join(ents['organizations'][:2])}" if ents["organizations"] else ""
        flags.append(Flag("Contact identity mismatch", 15,
                          f"Recruiting from a free webmail address{claim}", free))
    elif ents["organizations"] and ents["email_domains"]:
        org_keys = [_norm(o) for o in ents["organizations"]]
        unmatched = [d for d in ents["email_domains"]
                     if not any(k[:5] and k[:5] in _norm(d) for k in org_keys)]
        if unmatched:
            flags.append(Flag("Contact identity mismatch", 8,
                              "Email domain does not resemble any company named in the posting", unmatched))
    short = [u for u in ents["urls"] if u in SHORTENERS]
    if short:
        flags.append(Flag("Hidden link destination", 8, "Shortened URL hides where it leads", short))

    hourly = _hourly_estimate(text)
    if hourly and hourly >= 45 and any(k in low for k in ["no experience", "entry level", "entry-level", "data entry", "assistant"]):
        flags.append(Flag("Too good to be true", 12,
                          f"Pay works out to ≈ ${hourly:,.0f}/hr for an entry-level or no-experience role",
                          ents["money"][:3] or [f"${hourly:,.0f}/hr"]))

    # 3. Sentiment & hype
    sent = _sentiment(text)
    words = [t.text for t in doc if t.is_alpha]
    caps = [w for w in words if len(w) > 2 and w.isupper()]
    excl = text.count("!")
    caps_ratio = len(caps) / max(1, len(words))
    hype = []
    if excl >= 3:
        hype.append(f"{excl} exclamation marks")
    if caps_ratio > 0.05:
        hype.append(f"{caps_ratio:.0%} of words in ALL CAPS")
    if sent["compound"] > 0.9 and sent["pos"] > 0.25:
        hype.append(f"very high positive sentiment (pos {sent['pos']:.2f})")
    if hype:
        flags.append(Flag("Hype / emotional pressure", min(15, 5 * len(hype)), "; ".join(hype), caps[:5]))

    # 4. Pragmatics: missing expected sections
    present = {sec: any(c in low for c in cues) for sec, cues in EXPECTED_SECTIONS.items()}
    missing = [s for s, ok in present.items() if not ok]
    if len(missing) >= 2:
        flags.append(Flag("Missing normal posting content", 4 * len(missing),
                          "A legitimate posting usually explains these", missing))

    # 5. Style signals (from the stylometric profiler)
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    lengths = [sum(t.is_alpha for t in s) for s in sents]
    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
    style = {
        "sentences": len(lengths),
        "burstiness": burstiness(lengths) if lengths else 0.0,
        "mtld": mtld(lemmas),
        "second_person_rate": sum(t.lower_ in {"you", "your", "yours"} for t in doc) / max(1, len(words)),
    }
    if style["second_person_rate"] > 0.06 and len(words) > 40:
        flags.append(Flag("Hype / emotional pressure", 4,
                          f"Heavy direct address ('you/your' = {style['second_person_rate']:.0%} of words)", []))

    score = min(100.0, sum(f.points for f in flags))
    band = "High" if score >= 50 else "Elevated" if score >= 20 else "Low"
    return {
        "score": round(score),
        "band": band,
        "flags": [asdict(f) for f in sorted(flags, key=lambda f: -f.points)],
        "entities": ents,
        "sentiment": sent,
        "sections_present": present,
        "style": style,
        "intent_hits": hits,
    }


def highlight_spans(text: str, result: dict) -> list[tuple[int, int, str]]:
    """Character spans (start, end, category) for UI highlighting."""
    spans = []
    low = text.lower()
    for f in result["flags"]:
        # ALL-CAPS evidence must match its exact case; phrase cues match case-insensitively
        exact = f["category"] == "Hype / emotional pressure"
        hay = text if exact else low
        for ev in f["evidence"]:
            needle = str(ev) if exact else str(ev).lower()
            if len(needle) < 3:
                continue
            start = hay.find(needle)
            while start != -1:
                spans.append((start, start + len(needle), f["category"]))
                start = hay.find(needle, start + 1)
    spans.sort()
    merged = []
    for s in spans:  # drop overlaps, keep first
        if not merged or s[0] >= merged[-1][1]:
            merged.append(s)
    return merged
