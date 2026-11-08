"""
Interactive Stylometric & Syntactic Complexity Profiler
Run:  streamlit run app.py
"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from profiler.core import RADAR_METRICS, profile, radar_vector

# ---- palette (validated categorical slots 1-2 + neutral ink) ----
A_COLOR, B_COLOR = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ = [[0, "#f1f6fd"], [1, "#1c5aa6"]]  # single-hue sequential (blue)

SAMPLES = {
    "Lincoln — Gettysburg Address (1863)": (
        "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
        "conceived in Liberty, and dedicated to the proposition that all men are created equal. "
        "Now we are engaged in a great civil war, testing whether that nation, or any nation so "
        "conceived and so dedicated, can long endure. We are met on a great battle-field of that war. "
        "We have come to dedicate a portion of that field, as a final resting place for those who here "
        "gave their lives that that nation might live. It is altogether fitting and proper that we "
        "should do this."
    ),
    "Dickens — A Tale of Two Cities (1859)": (
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age "
        "of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season "
        "of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, "
        "we had everything before us, we had nothing before us, we were all going direct to Heaven, we "
        "were all going direct the other way. In short, the period was so far like the present period, "
        "that some of its noisiest authorities insisted on its being received, for good or for evil, in "
        "the superlative degree of comparison only."
    ),
    "Plain modern prose (sample)": (
        "The server went down at noon. Nobody noticed at first. Then the tickets started coming in, "
        "slowly and then all at once. We checked the logs. The disk was full. Someone had left debug "
        "logging on after Friday's release, and it had written forty gigabytes over the weekend. We "
        "cleared it, restarted the service, and wrote a rule so it would never happen again."
    ),
}

st.set_page_config(page_title="Job Posting Risk Check & Stylometric Profiler", page_icon="🔎", layout="wide")


def style(fig, h=360):
    fig.update_layout(
        height=h, margin=dict(l=40, r=20, t=80, b=40), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=INK_2, size=12), hovermode="closest",
        title=dict(y=0.97, yanchor="top", font=dict(color=INK, size=15)),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


@st.cache_data(show_spinner="Parsing…")
def run(text):
    p = profile(text)
    # dataclasses -> dicts so Streamlit can cache them
    p["syntax"]["sentences"] = [s.__dict__ for s in p["syntax"]["sentences"]]
    return p


# ================= mode switch =================
mode = st.sidebar.radio("Mode", ["Job posting risk", "Style profiler"])
if mode == "Job posting risk":
    from risk_page import render_risk_page
    render_risk_page(style)
    st.stop()

# ---------------- sidebar ----------------
st.sidebar.title("Input")
compare = st.sidebar.toggle("Compare two texts", value=True)
keys = list(SAMPLES)
a_key = st.sidebar.selectbox("Text A sample", keys, index=0)
text_a = st.sidebar.text_area("Text A", SAMPLES[a_key], height=220)
text_b = None
if compare:
    b_key = st.sidebar.selectbox("Text B sample", keys, index=1)
    text_b = st.sidebar.text_area("Text B", SAMPLES[b_key], height=220)

texts = {"A": text_a} | ({"B": text_b} if text_b else {})
profiles = {k: run(v) for k, v in texts.items() if v and v.strip()}
colors = {"A": A_COLOR, "B": B_COLOR}

st.title("Stylometric & Syntactic Complexity Profiler")
if not profiles:
    st.info("Paste some text in the sidebar.")
    st.stop()

tab_over, tab_syn, tab_lex, tab_rhy, tab_data = st.tabs(
    ["Overview", "Syntax trees", "Lexicon", "Rhythm", "Data"]
)

# ---------------- overview ----------------
with tab_over:
    for k, p in profiles.items():
        st.markdown(f"**Text {k}**")
        s, l, r = p["syntax"]["summary"], p["lexical"]["summary"], p["rhythm"]["summary"]
        c = st.columns(6)
        c[0].metric("Words / sentence", f"{s['mean_sentence_length']:.1f}")
        c[1].metric("Tree depth", f"{s['mean_tree_depth']:.2f}")
        c[2].metric("Dep. distance", f"{s['mean_dep_distance']:.2f}")
        c[3].metric("MTLD", f"{l['mtld']:.0f}")
        c[4].metric("FK grade", f"{l['fk_grade']:.1f}")
        c[5].metric("Burstiness", f"{r['burstiness']:+.2f}")

    fig = go.Figure()
    labels = list(RADAR_METRICS)
    for k, p in profiles.items():
        v = radar_vector(p)
        fig.add_trace(go.Scatterpolar(
            r=[v[x] for x in labels] + [v[labels[0]]], theta=labels + [labels[0]],
            name=f"Text {k}", line=dict(color=colors[k], width=2), fill="toself", opacity=0.55,
            hovertemplate="%{theta}: %{r:.2f}<extra>Text " + k + "</extra>",
        ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], gridcolor=GRID, showticklabels=False),
                                 angularaxis=dict(gridcolor=GRID)))
    st.plotly_chart(style(fig, 460), use_container_width=True)
    st.caption("Each axis is min–max normalised to a typical English range (see RADAR_METRICS in core.py).")

# ---------------- syntax ----------------
def dep_tree_figure(tree, color):
    """Linear-order layout: x = word position, y = depth below root. Arcs show head→dependent."""
    nodes, edges = [], []

    def walk(n, depth, parent=None):
        nodes.append((n["i"], depth, n["text"], n["pos"], n["dep"]))
        if parent is not None:
            edges.append((parent, (n["i"], depth)))
        for ch in n["children"]:
            walk(ch, depth + 1, (n["i"], depth))

    walk(tree, 0)
    fig = go.Figure()
    for (x0, y0), (x1, y1) in edges:
        fig.add_trace(go.Scatter(x=[x0, x1], y=[-y0, -y1], mode="lines",
                                 line=dict(color="#b8b7b1", width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[n[0] for n in nodes], y=[-n[1] for n in nodes], mode="markers+text",
        text=[n[2] for n in nodes], textposition="bottom center", textfont=dict(color=INK, size=12),
        marker=dict(size=12, color=color, line=dict(color="white", width=2)),
        customdata=[[n[3], n[4], n[1]] for n in nodes],
        hovertemplate="<b>%{text}</b><br>POS %{customdata[0]}<br>dep %{customdata[1]}<br>depth %{customdata[2]}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(title="depth", tickvals=[-d for d in range(12)], ticktext=list(range(12)))
    return fig


with tab_syn:
    which = st.radio("Text", list(profiles), horizontal=True, key="syn_text")
    sents = profiles[which]["syntax"]["sentences"]
    df = pd.DataFrame([{k: v for k, v in s.items() if k != "tree"} for s in sents])
    df.index = df.index + 1

    fig = go.Figure()
    for col, name, dash in [("tree_depth", "Tree depth", "solid"),
                            ("mean_dep_distance", "Mean dep. distance", "dot"),
                            ("mean_yngve", "Mean Yngve depth", "dash")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, mode="lines+markers",
                                 line=dict(color=colors[which], width=2, dash=dash), marker=dict(size=8)))
    fig.update_xaxes(title="sentence #", dtick=1)
    fig.update_layout(title="Complexity across the text", hovermode="x unified")
    st.plotly_chart(style(fig), use_container_width=True)

    idx = st.slider("Inspect sentence", 1, len(sents), 1)
    s = sents[idx - 1]
    st.markdown(f"> {s['text']}")
    c = st.columns(5)
    c[0].metric("Depth", s["tree_depth"]); c[1].metric("MDD", s["mean_dep_distance"])
    c[2].metric("Max Yngve", s["max_yngve"]); c[3].metric("Clauses", s["n_clauses"])
    c[4].metric("Right-branching", f"{s['right_branch_ratio']:.0%}")
    st.plotly_chart(style(dep_tree_figure(s["tree"], colors[which]), 420), use_container_width=True)

# ---------------- lexicon ----------------
with tab_lex:
    rows = []
    for k, p in profiles.items():
        l = p["lexical"]["summary"]
        rows.append({"text": k, **{m: l[m] for m in ["ttr", "mattr", "hdd", "lexical_density"]}})
    ldf = pd.DataFrame(rows).melt(id_vars="text", var_name="metric", value_name="value")
    fig = go.Figure()
    for k in profiles:
        d = ldf[ldf.text == k]
        fig.add_trace(go.Bar(x=d.metric, y=d.value, name=f"Text {k}", marker_color=colors[k],
                             marker_line=dict(color="white", width=2),
                             hovertemplate="%{x}: %{y:.3f}<extra>Text " + k + "</extra>"))
    fig.update_layout(title="Diversity & density (0–1 scale)", barmode="group", bargap=0.35)
    st.plotly_chart(style(fig), use_container_width=True)

    # Zipf plot: log rank vs log frequency; slope ≈ -1 for natural text
    fig = go.Figure()
    for k, p in profiles.items():
        freqs = [f for _, f in p["lexical"]["frequencies"]]
        words = [w for w, _ in p["lexical"]["frequencies"]]
        fig.add_trace(go.Scatter(x=list(range(1, len(freqs) + 1)), y=freqs, mode="markers",
                                 name=f"Text {k}", marker=dict(size=8, color=colors[k], line=dict(color="white", width=1)),
                                 text=words, hovertemplate="%{text}: rank %{x}, freq %{y}<extra></extra>"))
    fig.update_xaxes(type="log", title="rank (log)"); fig.update_yaxes(type="log", title="frequency (log)")
    fig.update_layout(title="Zipf distribution")
    st.plotly_chart(style(fig), use_container_width=True)

    st.dataframe(pd.DataFrame({k: p["lexical"]["summary"] for k, p in profiles.items()}).round(3),
                 use_container_width=True)

# ---------------- rhythm ----------------
with tab_rhy:
    fig = go.Figure()
    for k, p in profiles.items():
        L = p["rhythm"]["sentence_lengths"]
        fig.add_trace(go.Bar(x=list(range(1, len(L) + 1)), y=L, name=f"Text {k}", marker_color=colors[k],
                             marker_line=dict(color="white", width=2),
                             hovertemplate="sentence %{x}: %{y} words<extra>Text " + k + "</extra>"))
    fig.update_layout(title="Cadence — sentence length sequence", barmode="group")
    fig.update_xaxes(title="sentence #", dtick=1); fig.update_yaxes(title="words")
    st.plotly_chart(style(fig), use_container_width=True)

    which_r = st.radio("Stress map for", list(profiles), horizontal=True, key="rhy_text")
    pats = profiles[which_r]["rhythm"]["stress"]
    width = max(map(len, pats))
    z = [p + [math.nan] * (width - len(p)) for p in pats]
    fig = go.Figure(go.Heatmap(z=z, colorscale=SEQ, showscale=False, xgap=2, ygap=2,
                               hovertemplate="sentence %{y}, syllable %{x}: %{z}<extra></extra>",
                               y=list(range(1, len(z) + 1))))
    fig.update_layout(title="Approximate stress map (dark = stressed syllable)")
    fig.update_yaxes(autorange="reversed", title="sentence #", dtick=1); fig.update_xaxes(title="syllable")
    st.plotly_chart(style(fig, 160 + 30 * len(z)), use_container_width=True)

    st.dataframe(pd.DataFrame({k: p["rhythm"]["summary"] for k, p in profiles.items()}).round(3),
                 use_container_width=True)

# ---------------- raw data ----------------
with tab_data:
    for k, p in profiles.items():
        st.markdown(f"**Text {k} — per-sentence syntax**")
        st.dataframe(pd.DataFrame([{x: y for x, y in s.items() if x != "tree"} for s in p["syntax"]["sentences"]]),
                     use_container_width=True)
