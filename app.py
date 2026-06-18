import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

st.set_page_config(
    page_title="SentiScope · Social Informatics",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=DM+Mono:wght@400&display=swap');

:root {
    --bg:      #faf7f2;
    --card:    #ffffff;
    --border:  #e8e0d5;
    --accent:  #4a6fa5;
    --pos:     #5a9e6f;
    --neg:     #d96b6b;
    --neu:     #d4a843;
    --text:    #2d2d2d;
    --muted:   #8a8078;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
footer { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 700px !important; }

textarea {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
}
textarea:focus { border-color: var(--accent) !important; }

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.92rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* pill buttons - lighter style */
div[data-testid="column"] .stButton > button {
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 2px solid var(--border) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    opacity: 1 !important;
}

/* main analyse button */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stButton"]:last-child) .stButton > button {
    background: var(--accent) !important;
    color: white !important;
    width: 100%;
    padding: 0.65rem 2rem !important;
}

/* progress bars */
.stProgress > div > div { border-radius: 99px !important; }

/* metric */
[data-testid="stMetric"] {
    background: var(--bg);
    border: 2px solid var(--border);
    border-radius: 14px;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] { font-family: 'Nunito', sans-serif !important; font-size: 0.72rem !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; font-size: 1.1rem !important; color: var(--text) !important; }

/* expander */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem">
<div style="font-size:3.2rem;font-weight:900;color:#2d2d2d;margin-bottom:0.3rem">
        Senti<span style="color:#4a6fa5">Scope</span> ✦
    </div>

</div>
""", unsafe_allow_html=True)

# ── Mascot row ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:center;gap:1.2rem;margin:1.2rem 0 1.8rem;flex-wrap:wrap">
    <div style="background:white;border:2px solid #e8e0d5;border-radius:20px;padding:1rem 1.1rem;text-align:center;width:130px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
        <div style="font-size:2.5rem;line-height:1;margin-bottom:0.3rem">🥰</div>
        <div style="font-size:0.72rem;font-weight:900;letter-spacing:0.07em;text-transform:uppercase;color:#5a9e6f;margin-bottom:0.25rem">Positive</div>
        <div style="font-size:0.7rem;color:#8a8078;font-style:italic;line-height:1.4">"This is absolutely amazing!"</div>
    </div>
    <div style="background:white;border:2px solid #e8e0d5;border-radius:20px;padding:1rem 1.1rem;text-align:center;width:130px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
        <div style="font-size:2.5rem;line-height:1;margin-bottom:0.3rem">😐</div>
        <div style="font-size:0.72rem;font-weight:900;letter-spacing:0.07em;text-transform:uppercase;color:#d4a843;margin-bottom:0.25rem">Neutral</div>
        <div style="font-size:0.7rem;color:#8a8078;font-style:italic;line-height:1.4">"It was okay, I guess."</div>
    </div>
    <div style="background:white;border:2px solid #e8e0d5;border-radius:20px;padding:1rem 1.1rem;text-align:center;width:130px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
        <div style="font-size:2.5rem;line-height:1;margin-bottom:0.3rem">😤</div>
        <div style="font-size:0.72rem;font-weight:900;letter-spacing:0.07em;text-transform:uppercase;color:#d96b6b;margin-bottom:0.25rem">Negative</div>
        <div style="font-size:0.7rem;color:#8a8078;font-style:italic;line-height:1.4">"Terrible — never again!"</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Example pills ─────────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.72rem;font-weight:800;color:#8a8078;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">Try an example</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("😍 Best experience ever!"):
        st.session_state.input_text = "Best experience ever!"
        st.rerun()
with c2:
    if st.button("😐 It was okay, nothing special."):
        st.session_state.input_text = "It was okay, nothing special."
        st.rerun()
with c3:
    if st.button("😤 Terrible service, never again."):
        st.session_state.input_text = "Terrible service, never again."
        st.rerun()

st.write("")

# ── Text input ────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Your review",
    value=st.session_state.input_text,
    placeholder="e.g. The food was fantastic and the staff were so friendly!",
    height=110,
)

analyse_btn = st.button("Analyse Sentiment →", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyse(text):
    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)
    # Use highest score among pos/neg/neu to determine label
    scores = {"Positive": s["pos"], "Negative": s["neg"], "Neutral": s["neu"]}
    label = max(scores, key=scores.get)
    meta = {
        "Positive": ("🥰", "#5a9e6f"),
        "Negative": ("😤", "#d96b6b"),
        "Neutral":  ("😐", "#d4a843"),
    }
    emoji, color = meta[label]
    return dict(label=label, emoji=emoji, color=color, **s)

if analyse_btn:
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        r = analyse(user_input.strip())

        st.write("")
        st.markdown(f"""
        <div style="background:white;border:2px solid #e8e0d5;border-radius:20px;padding:1.4rem 1.6rem;box-shadow:0 2px 12px rgba(0,0,0,0.05)">
            <div style="display:flex;align-items:center;gap:0.9rem;margin-bottom:1rem">
                <div style="font-size:3rem;line-height:1">{r['emoji']}</div>
                <div>
                    <div style="font-size:1.8rem;font-weight:900;color:{r['color']};line-height:1">{r['label']}</div>
                    <div style="font-size:0.8rem;color:#8a8078;margin-top:0.15rem">VADER compound: <code style="color:#2d2d2d;background:#f0ece6;padding:0.1rem 0.4rem;border-radius:4px">{r['compound']:+.3f}</code></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Score bars using native Streamlit
        st.markdown('<p style="font-size:0.72rem;font-weight:800;color:#8a8078;text-transform:uppercase;letter-spacing:0.1em">Score Breakdown</p>', unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 4])
        with col_l:
            st.markdown('<span style="font-size:0.8rem;font-weight:700;color:#5a9e6f">Positive</span>', unsafe_allow_html=True)
        with col_r:
            st.progress(r["pos"])

        col_l, col_r = st.columns([1, 4])
        with col_l:
            st.markdown('<span style="font-size:0.8rem;font-weight:700;color:#d4a843">Neutral</span>', unsafe_allow_html=True)
        with col_r:
            st.progress(r["neu"])

        col_l, col_r = st.columns([1, 4])
        with col_l:
            st.markdown('<span style="font-size:0.8rem;font-weight:700;color:#d96b6b">Negative</span>', unsafe_allow_html=True)
        with col_r:
            st.progress(r["neg"])

        st.write("")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Positive", f"{r['pos']:.3f}")
        m2.metric("Negative", f"{r['neg']:.3f}")
        m3.metric("Neutral",  f"{r['neu']:.3f}")
        m4.metric("Compound", f"{r['compound']:+.3f}")

        st.write("")

        # SI note
        si_map = {
            "Positive": "People expressing satisfaction engage more with platforms — a key insight in Social Informatics when studying human–technology interaction.",
            "Negative": "Negative feedback signals friction in the user–system relationship — exactly what Social Informatics investigates to improve digital experiences.",
            "Neutral":  "Neutral language often reflects transactional intent — important when modelling online behaviour in Social Informatics research.",
        }
        st.info(f"💡 {si_map[r['label']]}")

# ── How it works ──────────────────────────────────────────────────────────────
st.write("")
with st.expander("🔬 How this works"):
    st.markdown("""
    This demo uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — the same
    sentiment extractor integrated into our full ML pipeline.

    | Step | Method |
    |------|--------|
    | Feature extraction | TF-IDF (unigrams + bigrams) + VADER scores |
    | Best classifier | Logistic Regression (saga solver, balanced classes) |
    | Live demo engine | VADER real-time scoring |

    The **compound score** ranges from −1 (most negative) to +1 (most positive).
    Threshold of ±0.05 separates classes — matching the training convention.
    """)

st.markdown('<div style="text-align:center;font-size:0.72rem;color:#c0b8b0;padding-top:1.5rem;margin-top:1rem;border-top:2px solid #e8e0d5">Social Informatics Group Assignment · SentiScope · WIX3002</div>', unsafe_allow_html=True)
