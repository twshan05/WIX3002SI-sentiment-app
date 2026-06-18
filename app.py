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
@import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,400;0,700;0,800;0,900;1,400&family=DM+Mono:wght@400&display=swap');

:root {
    --cream: #f5f0e8;
    --card:  #fffdf9;
    --border:#d4c9b8;
    --blue:  #3a5f9e;
    --pos:   #4e9268;
    --neg:   #c95c5c;
    --neu:   #c49a2a;
    --text:  #2a2520;
    --muted: #8a7e72;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    color: var(--text) !important;
    background-color: #ede8df !important;
    background-image:
        radial-gradient(ellipse at 15% 20%, rgba(255,220,180,0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 80%, rgba(180,210,255,0.25) 0%, transparent 50%) !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
footer { visibility: hidden; }
.block-container { padding: 1.5rem 1.5rem 4rem !important; max-width: 660px !important; }

/* textarea */
textarea {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    box-shadow: 3px 3px 0px #c4b8a4 !important;
}
textarea:focus { border-color: var(--blue) !important; }

/* all buttons */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 50px !important;
    border: 2px solid var(--border) !important;
    background: var(--card) !important;
    color: var(--text) !important;
    box-shadow: 2px 2px 0px #c4b8a4 !important;
    transition: all 0.12s !important;
}
.stButton > button:hover {
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    transform: translateY(-1px) !important;
    box-shadow: 3px 3px 0px #c4b8a4 !important;
}

/* metric */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 2px 3px 0px #c4b8a4 !important;
    padding: 0.6rem 0.8rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    color: var(--text) !important;
}

/* progress */
.stProgress > div > div { border-radius: 99px !important; background: #ddd5c8 !important; }
.stProgress > div > div > div { border-radius: 99px !important; }

/* info */
[data-testid="stAlert"] {
    background: #eef4ff !important;
    border: 2px solid #b8ccee !important;
    border-radius: 16px !important;
    box-shadow: 2px 3px 0px #b8ccee !important;
}

/* expander */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 2px 3px 0px #c4b8a4 !important;
}

/* label */
label[data-testid="stWidgetLabel"] p {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align:center;font-size:3.2rem;font-weight:900;color:#2a2520;"
    "letter-spacing:-0.03em;margin-bottom:0.2rem;font-family:Nunito,sans-serif'>"
    "Senti<span style='color:#3a5f9e'>Scope</span> ✦</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#8a7e72;font-style:italic;margin-bottom:0.5rem'>"
    "Type a review · discover its emotional tone</p>",
    unsafe_allow_html=True
)

st.divider()

# ── Mascot cards using st.columns ──────────────────────────────────────────
mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown("""
    <div style='background:#fffdf9;border:2px solid #d4c9b8;border-radius:20px;
    padding:1rem 0.8rem;text-align:center;box-shadow:3px 4px 0 #c4b8a4'>
        <div style='font-size:0.6rem;font-weight:900;color:#4e9268;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:0.5rem;background:#d4edda;border-radius:50px;
        padding:2px 8px;display:inline-block'>POSITIVE</div>
        <div style='font-size:2.5rem;line-height:1;margin:0.3rem 0'>🥰</div>
        <div style='font-size:0.7rem;color:#8a7e72;font-style:italic;line-height:1.4'>
        "This is absolutely amazing!"</div>
    </div>
    """, unsafe_allow_html=True)
with mc2:
    st.markdown("""
    <div style='background:#fffdf9;border:2px solid #d4c9b8;border-radius:20px;
    padding:1rem 0.8rem;text-align:center;box-shadow:3px 4px 0 #c4b8a4'>
        <div style='font-size:0.6rem;font-weight:900;color:#c49a2a;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:0.5rem;background:#fef9e7;border-radius:50px;
        padding:2px 8px;display:inline-block'>NEUTRAL</div>
        <div style='font-size:2.5rem;line-height:1;margin:0.3rem 0'>😐</div>
        <div style='font-size:0.7rem;color:#8a7e72;font-style:italic;line-height:1.4'>
        "It was okay, I guess."</div>
    </div>
    """, unsafe_allow_html=True)
with mc3:
    st.markdown("""
    <div style='background:#fffdf9;border:2px solid #d4c9b8;border-radius:20px;
    padding:1rem 0.8rem;text-align:center;box-shadow:3px 4px 0 #c4b8a4'>
        <div style='font-size:0.6rem;font-weight:900;color:#c95c5c;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:0.5rem;background:#fdecea;border-radius:50px;
        padding:2px 8px;display:inline-block'>NEGATIVE</div>
        <div style='font-size:2.5rem;line-height:1;margin:0.3rem 0'>😤</div>
        <div style='font-size:0.7rem;color:#8a7e72;font-style:italic;line-height:1.4'>
        "Terrible — never again!"</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Example pills ──────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-size:0.72rem;font-weight:900;color:#8a7e72;text-transform:uppercase;"
    "letter-spacing:0.12em;margin-bottom:0.3rem'>✏️ Try an example</p>",
    unsafe_allow_html=True
)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("😍 Best experience ever!"):
        st.session_state.input_text = "Best experience ever!"
        st.rerun()
with c2:
    if st.button("😐 Nothing special."):
        st.session_state.input_text = "It was okay, nothing special."
        st.rerun()
with c3:
    if st.button("😤 Never again."):
        st.session_state.input_text = "Terrible service, never again."
        st.rerun()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "YOUR REVIEW",
    value=st.session_state.input_text,
    placeholder="e.g. The food was fantastic and the staff were so friendly!",
    height=110,
)

analyse_btn = st.button("Analyse Sentiment →", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────
def analyse(text):
    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)
    scores = {"Positive": s["pos"], "Negative": s["neg"], "Neutral": s["neu"]}
    label = max(scores, key=scores.get)
    meta = {
        "Positive": ("🥰", "#4e9268", "#d4edda", "#a8d5b5"),
        "Negative": ("😤", "#c95c5c", "#fdecea", "#e8b0b0"),
        "Neutral":  ("😐", "#c49a2a", "#fef9e7", "#e8d090"),
    }
    emoji, color, bg, border = meta[label]
    return dict(label=label, emoji=emoji, color=color, bg=bg, border=border, **s)

if analyse_btn:
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        r = analyse(user_input.strip())
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Result header card
        st.markdown(
            f"<div style='background:{r['bg']};border:2px solid {r['border']};"
            f"border-radius:20px;padding:1.2rem 1.5rem;box-shadow:3px 4px 0 {r['border']};margin-bottom:0.8rem'>"
            f"<span style='font-size:2.8rem;vertical-align:middle'>{r['emoji']}</span>"
            f"&nbsp;&nbsp;<span style='font-size:1.9rem;font-weight:900;color:{r['color']};"
            f"font-family:Nunito,sans-serif;vertical-align:middle'>{r['label']}</span>"
            f"&nbsp;&nbsp;<code style='font-size:0.8rem;background:rgba(255,255,255,0.7);"
            f"padding:3px 8px;border-radius:8px;color:#2a2520'>compound {r['compound']:+.3f}</code>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Bars
        st.markdown(
            "<p style='font-size:0.7rem;font-weight:900;color:#8a7e72;text-transform:uppercase;"
            "letter-spacing:0.12em;margin-bottom:0.2rem'>Score Breakdown</p>",
            unsafe_allow_html=True
        )
        bl, br = st.columns([1, 4])
        with bl: st.markdown("<span style='font-size:0.82rem;font-weight:800;color:#4e9268'>Positive</span>", unsafe_allow_html=True)
        with br: st.progress(r["pos"])

        bl, br = st.columns([1, 4])
        with bl: st.markdown("<span style='font-size:0.82rem;font-weight:800;color:#c49a2a'>Neutral</span>", unsafe_allow_html=True)
        with br: st.progress(r["neu"])

        bl, br = st.columns([1, 4])
        with bl: st.markdown("<span style='font-size:0.82rem;font-weight:800;color:#c95c5c'>Negative</span>", unsafe_allow_html=True)
        with br: st.progress(r["neg"])

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Positive",  f"{r['pos']:.3f}")
        m2.metric("Negative",  f"{r['neg']:.3f}")
        m3.metric("Neutral",   f"{r['neu']:.3f}")
        m4.metric("Compound",  f"{r['compound']:+.3f}")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        si_map = {
            "Positive": "😊 People expressing satisfaction engage more with platforms — a key insight in Social Informatics when studying human–technology interaction.",
            "Negative": "💬 Negative feedback signals friction in the user–system relationship — exactly what Social Informatics investigates to improve digital experiences.",
            "Neutral":  "🔍 Neutral language often reflects transactional intent — important when modelling online behaviour in Social Informatics research.",
        }
        st.info(si_map[r['label']])

# ── How it works ───────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
with st.expander("🔬 How this works"):
    st.markdown("""
    This demo uses **VADER** — the same sentiment extractor in our full ML pipeline.

    | Step | Method |
    |------|--------|
    | Feature extraction | TF-IDF (unigrams + bigrams) + VADER scores |
    | Best classifier | Logistic Regression (saga, balanced classes) |
    | Live demo | VADER real-time scoring |

    Label = whichever of **pos / neg / neu** scores highest.
    """)

st.markdown(
    "<p style='text-align:center;font-size:0.7rem;color:#b0a898;font-style:italic;margin-top:2rem'>"
    "Social Informatics Group Assignment · SentiScope · WIX3002</p>",
    unsafe_allow_html=True
)
