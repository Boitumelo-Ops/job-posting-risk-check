"""Streamlit page: job-posting scam risk, with every flag traced back to an NLP step."""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from profiler.risk import assess, highlight_spans
from profiler.samples import LEGIT_POSTING, SCAM_POSTING

BAR = "#2a78d6"
STATUS = {  # reserved status colors, always paired with icon + label
    "Low": ("#0ca30c", "✓"),
    "Elevated": ("#ec835a", "!"),
    "High": ("#d03b3b", "✕"),
}
HIGHLIGHT = "#fde3c4"

EXAMPLES = {
    "Example: suspicious posting": SCAM_POSTING,
    "Example: normal posting": LEGIT_POSTING,
    "Paste your own": "",
}


@st.cache_data(show_spinner="Analysing posting…")
def _assess(text):
    return assess(text)


def _highlighted_html(text, result):
    out, pos = [], 0
    for start, end, cat in highlight_spans(text, result):
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark title="{html.escape(cat)}" style="background:{HIGHLIGHT};color:inherit;'
                   f'border-bottom:2px solid #c96a2b;padding:0 2px;border-radius:3px">'
                   f'{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return ('<div style="white-space:pre-wrap;line-height:1.7;font-size:15px;border:1px solid #e6e5e1;'
            'border-radius:8px;padding:16px">' + "".join(out) + "</div>")


def render_risk_page(style):
    st.sidebar.title("Posting")
    choice = st.sidebar.selectbox("Start from", list(EXAMPLES))
    text = st.sidebar.text_area("Job posting text", EXAMPLES[choice], height=380,
                                placeholder="Paste the full job posting, including contact details.")

    st.title("Job Posting Risk Check")
    st.caption("Rule-based NLP triage. A high score means *check carefully*, not *proven scam*; "
               "a low score is not a guarantee. Always verify the company independently.")
    if not text.strip():
        st.info("Paste a job posting in the sidebar.")
        return

    r = _assess(text)
    color, icon = STATUS[r["band"]]

    tab_result, tab_nlp = st.tabs(["Result", "How the NLP works"])

    with tab_result:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f"""<div style="border:1px solid #e6e5e1;border-radius:10px;padding:18px 20px">
                <div style="color:#52514e;font-size:13px">Risk score</div>
                <div style="font-size:56px;font-weight:600;line-height:1.1;color:#0b0b0b">{r['score']}
                <span style="font-size:18px;color:#52514e">/ 100</span></div>
                <div style="margin-top:8px;display:inline-flex;align-items:center;gap:8px;font-weight:600;color:#0b0b0b">
                  <span style="display:inline-flex;width:22px;height:22px;border-radius:50%;background:{color};
                   color:white;align-items:center;justify-content:center;font-size:13px">{icon}</span>
                  {r['band']} risk</div>
                <div style="height:6px;background:#eeede9;border-radius:3px;margin-top:14px">
                  <div style="height:6px;width:{r['score']}%;background:{color};border-radius:3px"></div></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            if r["flags"]:
                agg = (pd.DataFrame(r["flags"]).groupby("category")["points"].sum()
                       .sort_values())
                fig = go.Figure(go.Bar(
                    x=agg.values, y=agg.index, orientation="h", marker_color=BAR,
                    marker_line=dict(color="white", width=2),
                    text=[f"{v:.0f}" for v in agg.values], textposition="outside",
                    textfont=dict(color="#52514e"),
                    hovertemplate="%{y}: %{x:.0f} points<extra></extra>",
                ))
                fig.update_layout(title="What drives the score", bargap=0.35)
                fig.update_xaxes(title="points (total capped at 100)")
                st.plotly_chart(style(fig, 90 + 38 * len(agg)), use_container_width=True)
            else:
                st.success("No red-flag patterns found.")

        st.subheader("Highlighted posting")
        st.caption("Hover a highlight to see which check it triggered.")
        st.markdown(_highlighted_html(text, r), unsafe_allow_html=True)

        st.subheader("Why")
        for f in r["flags"]:
            with st.expander(f"**{f['category']}** · +{f['points']:.0f}"):
                st.write(f["detail"])
                if f["evidence"]:
                    st.markdown(" ".join(f"`{e}`" for e in f["evidence"]))

    with tab_nlp:
        e = r["entities"]
        st.markdown("#### 1 · Named Entity Recognition")
        st.caption("spaCy's statistical NER plus regular expressions for contact details. "
                   "The small model makes mistakes; compare its output with the text yourself.")
        rows = [(k.replace("_", " ").title(), ", ".join(v)) for k, v in e.items() if v]
        st.table(pd.DataFrame(rows, columns=["Entity type", "Found"]).set_index("Entity type"))

        st.markdown("#### 2 · Intent analysis")
        st.caption("A PhraseMatcher scans the lowercased tokens for cue phrases grouped by scam intent.")
        if r["intent_hits"]:
            st.table(pd.DataFrame([(k, ", ".join(sorted(set(v)))) for k, v in r["intent_hits"].items()],
                                  columns=["Intent", "Cues"]).set_index("Intent"))
        else:
            st.write("No intent cues matched.")

        st.markdown("#### 3 · Sentiment analysis")
        st.caption("VADER, a lexicon-based sentiment model tuned for short informal text.")
        s = r["sentiment"]
        c = st.columns(4)
        c[0].metric("Positive", f"{s['pos']:.2f}"); c[1].metric("Neutral", f"{s['neu']:.2f}")
        c[2].metric("Negative", f"{s['neg']:.2f}"); c[3].metric("Compound", f"{s['compound']:+.2f}")

        st.markdown("#### 4 · Pragmatics: what a normal posting contains")
        for sec, ok in r["sections_present"].items():
            st.markdown(f"{'✓' if ok else '✕'} {sec} — {'present' if ok else '**missing**'}")

        st.markdown("#### 5 · Syntax & style (from the profiler)")
        st_ = r["style"]
        c = st.columns(3)
        c[0].metric("Sentences", st_["sentences"])
        c[1].metric("Burstiness", f"{st_['burstiness']:+.2f}")
        c[2].metric("'you/your' rate", f"{st_['second_person_rate']:.0%}")
