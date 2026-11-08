# Interactive Stylometric & Syntactic Complexity Profiler

Paste one or two texts and see how their style compares on three levels:

| Layer | Question it answers | Main metrics |
|---|---|---|
| **Syntax** | How deeply nested and hard to process are the sentences? | Tree depth, Mean Dependency Distance, Yngve depth, clauses/sentence, subordination index, branching direction |
| **Lexicon** | How varied and dense is the vocabulary? | TTR, Guiraud's R, MATTR, MTLD, HD-D, Yule's K, Honoré's R, lexical density, Flesch / FK grade |
| **Rhythm** | How does the prose move? | Sentence-length CV, burstiness, lag-1 autocorrelation, length entropy, nPVI, stress alternation |

It also has a **Job posting risk** mode (the default). It reads a job posting and scores its scam risk from 0 to 100. Every point is traced back to words in the text.

### Job posting risk: the NLP pipeline

| Trailhead concept | Implementation (`profiler/risk.py`) |
|---|---|
| Segmentation, tokenization, lemmatization, POS | the same spaCy pipeline the profiler uses |
| **Named Entity Recognition** | spaCy NER (ORG, PERSON, MONEY, GPE) plus regular expressions for emails, URLs and phone numbers |
| **Intent analysis** | a spaCy `PhraseMatcher` with six intent groups: upfront payment or fake check, requests for personal data, off-platform contact, urgency, too good to be true, vague or reshipping roles |
| **Sentiment analysis** | VADER polarity, plus hype markers: exclamation marks, ALL CAPS, heavy "you/your" |
| **Pragmatics / context** | checks whether the parts a real posting normally has are missing: responsibilities, qualifications, company identity, how to apply |
| NER consistency | free webmail or a domain unrelated to the named company; link shorteners; pay that is unrealistic for an entry-level role |
| Syntax / style | burstiness, MTLD and direct address from the profiler |

Scoring is transparent. Each flag adds weighted points up to a cap, and the total is capped at 100. Bands: under 20 is **Low**, 20–49 is **Elevated**, 50 and over is **High**.
The score is a triage aid, not a verdict.

**Next steps for this mode:** collect labeled postings (public scam reports alongside real listings) and measure precision and recall. Then train a classifier, for example logistic regression over these features plus TF-IDF, and compare it with the rule system. Add a WHOIS domain-age check and expose `assess()` through a FastAPI endpoint so Access Authenticity can call it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest -q                 # 14 tests should pass
streamlit run app.py      # opens http://localhost:8501
```

## Architecture

```
profiler/
  core.py     profile(text) -> {syntax, lexical, rhythm}; radar normalisation
  syntax.py   dependency-tree metrics (pure functions over spaCy Doc/Span)
  lexical.py  diversity, density, readability
  rhythm.py   cadence and approximate stress
  risk.py     job-posting scam risk: NER, intent, sentiment, pragmatics
  samples.py  fictional example postings
app.py        Streamlit + Plotly UI; mode switch
risk_page.py  Job posting risk page
tests/        unit tests with hand-computed expected values
```

The analysis package knows nothing about the UI. You can reuse it in a notebook, a FastAPI
endpoint, or a batch job over a whole corpus.

---

## The theory, layer by layer

### 1. Hierarchical sentence parsing

A **dependency parse** turns a sentence into a rooted tree. Each word (except the root verb)
points to the word it modifies, its *head*. *"The cat sat"* becomes `sat → cat → The`.

* **Tree depth** is the longest path from the root to a leaf: h(t) = 0 for a leaf, otherwise 1 + max over the children's h. Deeper trees mean more nested modification.
* **Mean Dependency Distance (MDD)** (Liu 2008) is the average of |position(word) − position(head)|. Languages tend to keep MDD small. A high MDD means the reader has to hold a word in working memory longer before its head arrives.
* **Yngve depth** (Yngve 1960) counts, at each word, how many constituents are still "promised" to the right. At each node the children are ordered left to right; taking branch *k* of *m* costs *m − 1 − k*. Sum these costs along the path from the root. Left-branching, front-loaded sentences score high. The original definition is over constituency trees; `syntax.py` adapts it to dependency trees.
* **Clauses & subordination index**: count clauses from the labels `advcl, ccomp, xcomp, csubj, acl, relcl` plus verbal `conj`. The index is subordinate clauses ÷ all clauses.
* **Right-branching ratio**: the share of dependents that come after their head.

**Formal-grammar extension:** for real phrase-structure (CFG) trees, add
[benepar](https://github.com/nikitakit/self-attentive-parser) as a spaCy component. You can then
compute true Yngve and Frazier scores and the number of NP/VP/SBAR nodes on constituency trees.

### 2. Lexical complexity

Let N be the number of tokens, V the number of types (distinct words), and V(m,N) the number of types that occur exactly m times.

* **TTR = V/N** falls as N grows. By Heaps' law, V ≈ K·N^β with β < 1, so plain TTR is not comparable across texts of different length.
* **MATTR** averages TTR over a sliding window of 50 tokens.
* **MTLD** gives the average length of stretches that keep TTR above 0.72. It is computed forwards and backwards and the two results are averaged.
* **HD-D** is the expected TTR of a random 42-token sample. It uses the hypergeometric probability that each type appears: 1 − C(N−f, 42)/C(N, 42).
* **Yule's K = 10⁴ · (Σ m²·V(m,N) − N) / N²** measures repetitiveness. It is almost independent of length.
* **Honoré's R = 100·log N / (1 − V₁/V)** rewards hapax legomena (words used only once).
* **Lexical density** is the share of words that are content words (nouns, verbs, adjectives, adverbs).

The measures are computed on **lemmas**, so *run/ran/running* count as one type.

### 3. Rhythm

* **Burstiness B = (σ − μ)/(σ + μ)** over sentence lengths. −1 means perfectly even, 0 is random (Poisson-like), and values near +1 mean long and short sentences come in clusters.
* **Lag-1 autocorrelation**: a positive value means long sentences follow long ones. A negative value means the text alternates long and short, a classic rhetorical cadence.
* **nPVI** (Grabe & Low 2002) = 100/(n−1) · Σ |dₖ − dₖ₊₁| / ((dₖ + dₖ₊₁)/2). Phoneticians compute it on vowel durations; here syllables per word stand in for duration.
* **Stress alternation** is the share of adjacent syllables that switch between stressed and unstressed.

⚠️ Stress here is a **heuristic**: content words get stress on their first syllable and function
words are unstressed. To use real lexical stress, `pip install pronouncing` and look words up in the CMU
Pronouncing Dictionary (its digits 0/1/2 mark stress level).

---

## Roadmap (suggested order)

1. **Constituency parsing** with benepar, plus a collapsible tree view.
2. **CMU stress** in place of the heuristic, plus detection of metrical feet (iamb, trochee, dactyl).
3. **Authorship attribution**: Burrows' Delta on the 150 most frequent words. Delta is the mean |z-score difference| between two texts. Show the result as a heatmap or dendrogram.
4. **Corpus mode**: upload many .txt files, then cluster authors with PCA or t-SNE over the feature vector.
5. **Deploy** to Streamlit Community Cloud or a small AWS instance.

## References
Liu, H. (2008) *Dependency distance as a metric of language comprehension difficulty.* ·
Yngve, V. (1960) *A model and an hypothesis for language structure.* ·
McCarthy & Jarvis (2010) *MTLD, vocd-D, and HD-D.* · Covington & McFall (2010) *MATTR.* ·
Goh & Barabási (2008) *Burstiness and memory in complex systems.* ·
Grabe & Low (2002) *Durational variability in speech and the rhythm class hypothesis.* ·
Burrows, J. (2002) *'Delta': a measure of stylistic difference.*
