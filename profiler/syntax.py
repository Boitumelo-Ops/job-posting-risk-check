"""
Syntactic complexity metrics over spaCy dependency trees.

A dependency parse is a rooted tree T = (V, E) where V is the set of tokens and
each edge (h -> d) links a head to a dependent. Every metric below is a
function of that tree's *shape* (depth, branching) or of its *linearisation*
(how far apart heads and dependents sit in the surface string).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

# Dependency labels that introduce a finite or non-finite subordinate clause.
SUBORDINATE_DEPS = {"advcl", "ccomp", "xcomp", "csubj", "csubjpass", "acl", "relcl"}
PUNCT = {"PUNCT", "SPACE"}


@dataclass
class SentenceSyntax:
    text: str
    n_tokens: int
    tree_depth: int            # longest root-to-leaf path (in edges)
    mean_dep_distance: float   # MDD (Liu 2008)
    mean_yngve: float          # Yngve (1960) left-branching memory load
    max_yngve: int
    n_clauses: int             # finite + non-finite clauses
    n_subordinate: int
    right_branch_ratio: float  # share of dependents that follow their head
    tree: dict = field(repr=False, default_factory=dict)  # nested structure for viz


def _content_tokens(sent):
    return [t for t in sent if t.pos_ not in PUNCT]


def tree_depth(token) -> int:
    """Height of the subtree rooted at `token`: h(t) = 0 if leaf else 1 + max h(child)."""
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(tree_depth(c) for c in children)


def mean_dependency_distance(sent) -> float:
    """
    MDD = (1 / (n - 1)) * sum_i |pos(w_i) - pos(head(w_i))|  over non-root, non-punct words.
    Liu (2008) shows natural languages minimise this; higher MDD = heavier working-memory load.
    """
    dists = [abs(t.i - t.head.i) for t in sent if t.head.i != t.i and t.pos_ not in PUNCT]
    return mean(dists) if dists else 0.0


def yngve_scores(sent) -> list[int]:
    """
    Yngve depth adapted to dependency trees.
    Walk from the root: at each node, order its children left-to-right (including the
    node itself as the 'head slot'). Branch k of m receives a cost of (m - 1 - k): the
    number of constituents still 'promised' to the right that must be held in memory.
    A word's Yngve score is the sum of costs along its path from the root.
    """
    scores: dict[int, int] = {}

    def walk(node, acc):
        slots = sorted([node, *node.children], key=lambda t: t.i)
        m = len(slots)
        for k, s in enumerate(slots):
            cost = acc + (m - 1 - k)
            if s.i == node.i:
                scores[node.i] = cost
            else:
                walk(s, cost)

    walk(sent.root, 0)
    return [scores[t.i] for t in sent if t.pos_ not in PUNCT and t.i in scores]


def count_clauses(sent) -> tuple[int, int]:
    """Clauses = 1 (main) + verbal conjuncts + subordinate clauses."""
    sub = sum(1 for t in sent if t.dep_ in SUBORDINATE_DEPS)
    coord = sum(1 for t in sent if t.dep_ == "conj" and t.pos_ in {"VERB", "AUX"})
    return 1 + sub + coord, sub


def to_nested(token) -> dict:
    """Serialise subtree for visualisation."""
    return {
        "i": token.i,
        "text": token.text,
        "pos": token.pos_,
        "dep": token.dep_,
        "children": [to_nested(c) for c in token.children if c.pos_ not in PUNCT],
    }


def analyse_sentence(sent) -> SentenceSyntax:
    words = _content_tokens(sent)
    deps = [t for t in words if t.head.i != t.i]
    right = sum(1 for t in deps if t.i > t.head.i)
    yng = yngve_scores(sent)
    n_clauses, n_sub = count_clauses(sent)
    return SentenceSyntax(
        text=sent.text.strip(),
        n_tokens=len(words),
        tree_depth=tree_depth(sent.root),
        mean_dep_distance=round(mean_dependency_distance(sent), 3),
        mean_yngve=round(mean(yng), 3) if yng else 0.0,
        max_yngve=max(yng) if yng else 0,
        n_clauses=n_clauses,
        n_subordinate=n_sub,
        right_branch_ratio=round(right / len(deps), 3) if deps else 0.0,
        tree=to_nested(sent.root),
    )


def document_syntax(doc) -> dict:
    sents = [analyse_sentence(s) for s in doc.sents if _content_tokens(s)]
    if not sents:
        return {"sentences": [], "summary": {}}
    total_clauses = sum(s.n_clauses for s in sents)
    summary = {
        "mean_sentence_length": mean(s.n_tokens for s in sents),
        "mean_tree_depth": mean(s.tree_depth for s in sents),
        "mean_dep_distance": mean(s.mean_dep_distance for s in sents),
        "mean_yngve": mean(s.mean_yngve for s in sents),
        "clauses_per_sentence": total_clauses / len(sents),
        # Subordination index: dependent clauses / all clauses (Hunt's T-unit tradition)
        "subordination_index": sum(s.n_subordinate for s in sents) / total_clauses,
        "right_branch_ratio": mean(s.right_branch_ratio for s in sents),
    }
    return {"sentences": sents, "summary": summary}
