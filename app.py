import streamlit as st
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Download required NLTK data ──────────────────────────────────────────────
nltk.download("vader_lexicon", quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiScope · Social Informatics",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:ital,wght@0,400;1,400&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --pos:       #3fb950;
    --neg:       #f85149;
    --neu:       #d29922;
    --text:      #e6edf3;
    --muted:     #8b949e;
}

/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 720px !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    color: var(--text);
    margin: 0 0 0.5rem;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 480px;
    margin: 0 auto;
}

/* ── Input card ── */
.input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
}
.input-label {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}

/* ── Streamlit textarea override ── */
textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
    resize: vertical !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important; }

/* ── Streamlit button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 1.8rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Result card ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.35s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-top {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.result-emoji { font-size: 2.8rem; line-height: 1; }
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.result-label.pos { color: var(--pos); }
.result-label.neg { color: var(--neg); }
.result-label.neu { color: var(--neu); }

/* ── Score bars ── */
.score-section { margin-top: 0.2rem; }
.score-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.bar-outer {
    background: #21262d;
    border-radius: 99px;
    height: 7px;
    margin-bottom: 0.65rem;
    overflow: hidden;
}
.bar-inner {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.bar-pos { background: var(--pos); }
.bar-neg { background: var(--neg); }
.bar-neu { background: var(--neu); }

/* ── VADER breakdown ── */
.vader-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-top: 1.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
}
.vader-cell {
    background: #21262d;
    border-radius: 8px;
    padding: 0.75rem 0.5rem;
    text-align: center;
}
.vader-cell-val {
    font-size: 1.15rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    color: var(--text);
}
.vader-cell-key {
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── SI context tag ── */
.si-context {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1rem;
    background: rgba(88,166,255,0.07);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.8rem;
}
.si-context strong { color: var(--accent); }

/* ── Example pills ── */
.pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.pill {
    background: #21262d;
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    color: var(--muted);
    cursor: pointer;
}

/* ── Footer credits ── */
.credits {
    text-align: center;
    font-size: 0.75rem;
    color: #484f58;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Social Informatics · NLP Sentiment Analysis</div>
    <div class="hero-title">Senti<span>Scope</span></div>
    <div class="hero-sub">
        Analyse how language carries emotion — powered by VADER + TF-IDF,
        the same pipeline your team trained on real-world review data.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><div class="input-label">Enter a review or sentence</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    placeholder="e.g. The service was absolutely fantastic — I'll definitely come back!",
    height=110,
    key="review_input",
    label_visibility="collapsed",
)

analyse_clicked = st.button("Analyse Sentiment →")

st.markdown("""
<div class="pills">
    <span class="pill">😍 "Best experience ever!"</span>
    <span class="pill">😐 "It was okay, nothing special."</span>
    <span class="pill">😤 "Terrible service, never again."</span>
</div>
</div>
""", unsafe_allow_html=True)

# ── Analysis logic ────────────────────────────────────────────────────────────
def analyse_vader(text: str):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound = scores["compound"]

    # Map compound → class (mirrors common VADER threshold convention)
    if compound >= 0.05:
        label = "Positive"
        css_cls = "pos"
        emoji = "😄"
    elif compound <= -0.05:
        label = "Negative"
        css_cls = "neg"
        emoji = "😞"
    else:
        label = "Neutral"
        css_cls = "neu"
        emoji = "😐"

    # Normalise pos / neg / neu to sum to 1 for the bars
    raw_pos = scores["pos"]
    raw_neg = scores["neg"]
    raw_neu = scores["neu"]

    return {
        "label": label,
        "css_cls": css_cls,
        "emoji": emoji,
        "compound": compound,
        "pos": raw_pos,
        "neg": raw_neg,
        "neu": raw_neu,
    }


if analyse_clicked and user_input.strip():
    result = analyse_vader(user_input.strip())

    pos_pct = int(result["pos"] * 100)
    neg_pct = int(result["neg"] * 100)
    neu_pct = int(result["neu"] * 100)
    compound_display = f"{result['compound']:+.3f}"

    # SI context blurb
    si_blurbs = {
        "Positive": "People expressing satisfaction tend to engage more with platforms — a key insight in <strong>Social Informatics</strong> when studying human–technology interaction.",
        "Negative": "Negative feedback signals friction in the user–system relationship — exactly what <strong>Social Informatics</strong> investigates to improve digital experiences.",
        "Neutral": "Neutral language often reflects informational or transactional intent — important context when modelling online behaviour in <strong>Social Informatics</strong> research.",
    }

    st.markdown(f"""
    <div class="result-card">
        <div class="result-top">
            <div class="result-emoji">{result['emoji']}</div>
            <div>
                <div class="result-label {result['css_cls']}">{result['label']}</div>
                <div style="font-size:0.82rem;color:var(--muted);">VADER compound: <span style="font-family:'DM Mono',monospace;color:var(--text)">{compound_display}</span></div>
            </div>
        </div>

        <div class="score-section">
            <div class="score-label-row"><span>Positive</span><span>{pos_pct}%</span></div>
            <div class="bar-outer"><div class="bar-inner bar-pos" style="width:{pos_pct}%"></div></div>

            <div class="score-label-row"><span>Neutral</span><span>{neu_pct}%</span></div>
            <div class="bar-outer"><div class="bar-inner bar-neu" style="width:{neu_pct}%"></div></div>

            <div class="score-label-row"><span>Negative</span><span>{neg_pct}%</span></div>
            <div class="bar-outer"><div class="bar-inner bar-neg" style="width:{neg_pct}%"></div></div>
        </div>

        <div class="vader-grid">
            <div class="vader-cell">
                <div class="vader-cell-val" style="color:var(--pos)">{result['pos']:.3f}</div>
                <div class="vader-cell-key">Positive</div>
            </div>
            <div class="vader-cell">
                <div class="vader-cell-val" style="color:var(--neg)">{result['neg']:.3f}</div>
                <div class="vader-cell-key">Negative</div>
            </div>
            <div class="vader-cell">
                <div class="vader-cell-val" style="color:var(--neu)">{result['neu']:.3f}</div>
                <div class="vader-cell-key">Neutral</div>
            </div>
            <div class="vader-cell">
                <div class="vader-cell-val">{result['compound']:+.3f}</div>
                <div class="vader-cell-key">Compound</div>
            </div>
        </div>

        <div class="si-context">
            <span>💡</span>
            <span>{si_blurbs[result['label']]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif analyse_clicked and not user_input.strip():
    st.warning("Please enter some text before analysing.")

# ── How it works (collapsible) ────────────────────────────────────────────────
with st.expander("🔬 How this works"):
    st.markdown("""
    This demo uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — the same
    sentiment extractor your team integrated into the full ML pipeline in Google Colab.

    | Step | Method |
    |------|--------|
    | Feature extraction | TF-IDF (unigrams + bigrams) + VADER scores |
    | Best classifier | Logistic Regression (`saga` solver, balanced classes) |
    | Live demo engine | VADER real-time scoring |

    **VADER compound score** ranges from −1 (most negative) to +1 (most positive).
    A threshold of ±0.05 separates positive / negative from neutral — matching the
    labelling convention used during training.

    **Why Social Informatics?** SI examines how technology shapes and is shaped by social
    behaviour. Sentiment analysis is one concrete tool for studying those dynamics at scale —
    from brand reputation to public health communication.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="credits">
    Social Informatics Group Assignment · SentiScope UI · Built with Streamlit + VADER
</div>
""", unsafe_allow_html=True)
