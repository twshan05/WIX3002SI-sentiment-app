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
@import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,400;0,600;0,700;0,800;0,900;1,400;1,700&family=DM+Mono:wght@400&display=swap');

:root {
    --cream:   #f5f0e8;
    --card:    #fffdf9;
    --border:  #d4c9b8;
    --blue:    #3a5f9e;
    --pos:     #4e9268;
    --neg:     #c95c5c;
    --neu:     #c49a2a;
    --text:    #2a2520;
    --muted:   #8a7e72;
    --doodle:  #2a2520;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    color: var(--text) !important;
}

/* Textured background using SVG noise pattern */
[data-testid="stAppViewContainer"] {
    background-color: #f5f0e8 !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E"),
        radial-gradient(ellipse at 15% 20%, rgba(255,220,180,0.25) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 75%, rgba(180,210,255,0.2) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 50%, rgba(245,240,232,1) 0%, rgba(238,230,218,1) 100%);
    background-size: 400px 400px, 100% 100%, 100% 100%, 100% 100%;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
footer { visibility: hidden; }
.block-container { padding: 1.5rem 1.5rem 4rem !important; max-width: 680px !important; }

/* ── Wavy/sketchy card style ── */
.sketch-card {
    background: var(--card);
    border-radius: 24px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    box-shadow: 3px 4px 0px #c4b8a4, 0 1px 3px rgba(0,0,0,0.06);
    border: 2px solid var(--border);
}

/* ── Textarea ── */
textarea {
    background: #fffdf9 !important;
    border: 2px solid var(--border) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    box-shadow: 2px 3px 0px #c4b8a4 !important;
}
textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 2px 3px 0px var(--blue) !important;
    outline: none !important;
}

/* ── ALL buttons base ── */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 50px !important;
    transition: all 0.15s !important;
    border: 2px solid var(--border) !important;
}

/* ── Pill buttons (inside columns) ── */
div[data-testid="column"] .stButton > button {
    background: #fffdf9 !important;
    color: var(--text) !important;
    font-size: 0.8rem !important;
    padding: 0.45rem 0.8rem !important;
    box-shadow: 2px 2px 0px #c4b8a4 !important;
    width: 100%;
}
div[data-testid="column"] .stButton > button:hover {
    background: #fff8ee !important;
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    box-shadow: 2px 2px 0px var(--blue) !important;
    transform: translateY(-1px);
}
div[data-testid="column"] .stButton > button:active {
    transform: translateY(1px);
    box-shadow: 1px 1px 0px #c4b8a4 !important;
}

/* ── Main analyse button ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] .stButton > button {
    background: var(--blue) !important;
    color: white !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    border: 2px solid #2a4a7f !important;
    box-shadow: 3px 4px 0px #2a4a7f !important;
    letter-spacing: 0.02em;
    width: 100%;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] .stButton > button:hover {
    background: #2e4f8a !important;
    transform: translateY(-1px);
    box-shadow: 4px 5px 0px #2a4a7f !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] .stButton > button:active {
    transform: translateY(2px);
    box-shadow: 1px 2px 0px #2a4a7f !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    border-radius: 99px !important;
    background: #e0d8cc !important;
}
.stProgress > div > div > div {
    border-radius: 99px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #fffdf9 !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 0.7rem 0.8rem !important;
    box-shadow: 2px 3px 0px #c4b8a4 !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.68rem !important;
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

/* ── Info box ── */
[data-testid="stAlert"] {
    background: #eef4ff !important;
    border: 2px solid #b8ccee !important;
    border-radius: 16px !important;
    box-shadow: 2px 3px 0px #b8ccee !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #fffdf9 !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 2px 3px 0px #c4b8a4 !important;
}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── Hero with hand-drawn doodle feel ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;position:relative">

    <!-- decorative doodle stars -->
    <svg style="position:absolute;top:18px;left:40px;opacity:0.18" width="38" height="38" viewBox="0 0 38 38">
        <path d="M19 2 L21.5 14 L34 10 L24 19 L34 28 L21.5 24 L19 36 L16.5 24 L4 28 L14 19 L4 10 L16.5 14 Z"
              fill="none" stroke="#2a2520" stroke-width="1.8" stroke-linejoin="round"/>
    </svg>
    <svg style="position:absolute;top:10px;right:50px;opacity:0.13" width="28" height="28" viewBox="0 0 28 28">
        <path d="M14 2 L15.8 10 L24 8 L17.5 14 L24 20 L15.8 18 L14 26 L12.2 18 L4 20 L10.5 14 L4 8 L12.2 10 Z"
              fill="none" stroke="#2a2520" stroke-width="1.6" stroke-linejoin="round"/>
    </svg>
    <!-- small circle doodles -->
    <svg style="position:absolute;top:50px;left:80px;opacity:0.1" width="18" height="18" viewBox="0 0 18 18">
        <circle cx="9" cy="9" r="7" fill="none" stroke="#2a2520" stroke-width="1.5"/>
    </svg>
    <svg style="position:absolute;top:35px;right:100px;opacity:0.1" width="12" height="12" viewBox="0 0 12 12">
        <circle cx="6" cy="6" r="5" fill="none" stroke="#2a2520" stroke-width="1.3"/>
    </svg>

    <div style="font-size:3.4rem;font-weight:900;color:#2a2520;letter-spacing:-0.03em;line-height:1.1;margin-bottom:0.4rem">
        Senti<span style="color:#3a5f9e">Scope</span>
        <span style="font-size:2rem">✦</span>
    </div>
    <div style="font-size:0.88rem;color:#8a7e72;max-width:380px;margin:0 auto;line-height:1.65;font-style:italic">
        Type a review · discover its emotional tone
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mascot row ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:center;gap:1rem;margin:1.6rem 0 1.4rem;flex-wrap:wrap">

    <div style="background:#fffdf9;border:2px solid #d4c9b8;border-radius:22px;padding:1.1rem 1rem;text-align:center;width:136px;box-shadow:3px 4px 0px #c4b8a4;position:relative">
        <div style="position:absolute;top:-10px;right:-8px;font-size:0.65rem;background:#d4edda;border:1.5px solid #4e9268;border-radius:50px;padding:2px 7px;color:#4e9268;font-weight:800">POSITIVE</div>
        <div style="font-size:2.6rem;line-height:1;margin-bottom:0.4rem">🥰</div>
        <div style="font-size:0.7rem;color:#8a7e72;font-style:italic;line-height:1.4">"This is absolutely amazing!"</div>
    </div>

    <div style="background:#fffdf9;border:2px solid #d4c9b8;border-radius:22px;padding:1.1rem 1rem;text-align:center;width:136px;box-shadow:3px 4px 0px #c4b8a4;position:relative">
        <div style="position:absolute;top:-10px;right:-8px;font-size:0.65rem;background:#fef9e7;border:1.5px solid #c49a2a;border-radius:50px;padding:2px 7px;color:#c49a2a;font-weight:800">NEUTRAL</div>
        <div style="font-size:2.6rem;line-height:1;margin-bottom:0.4rem">😐</div>
        <div style="font-size:0.7rem;color:#8a7e72;font-style:italic;line-height:1.4">"It was okay, I guess."</div>
    </div>

    <div style="background:#fffdf9;border:2px solid #d4c9b8;border-radius:22px;padding:1.1rem 1rem;text-align:center;width:136px;box-shadow:3px 4px 0px #c4b8a4;position:relative">
        <div style="position:absolute;top:-10px;right:-8px;font-size:0.65rem;background:#fdecea;border:1.5px solid #c95c5c;border-radius:50px;padding:2px 7px;color:#c95c5c;font-weight:800">NEGATIVE</div>
        <div style="font-size:2.6rem;line-height:1;margin-bottom:0.4rem">😤</div>
        <div style="font-size:0.7rem;color:#8a7e72;font-style:italic;line-height:1.4">"Terrible — never again!"</div>
    </div>

</div>
""", unsafe_allow_html=True)

# ── Divider doodle ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin:0.2rem 0 1rem;opacity:0.2">
    <svg width="220" height="12" viewBox="0 0 220 12">
        <path d="M0 6 Q 27 1, 55 6 Q 82 11, 110 6 Q 137 1, 165 6 Q 192 11, 220 6"
              fill="none" stroke="#2a2520" stroke-width="1.8" stroke-linecap="round"/>
    </svg>
</div>
""", unsafe_allow_html=True)

# ── Example pills ─────────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.72rem;font-weight:900;color:#8a7e72;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem">✏️ Try an example</p>', unsafe_allow_html=True)

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

# ── Analysis logic ────────────────────────────────────────────────────────────
def analyse(text):
    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)
    scores = {"Positive": s["pos"], "Negative": s["neg"], "Neutral": s["neu"]}
    label = max(scores, key=scores.get)
    meta = {
        "Positive": ("🥰", "#4e9268", "#d4edda", "#4e9268"),
        "Negative": ("😤", "#c95c5c", "#fdecea", "#c95c5c"),
        "Neutral":  ("😐", "#c49a2a", "#fef9e7", "#c49a2a"),
    }
    emoji, color, bg, border = meta[label]
    return dict(label=label, emoji=emoji, color=color, bg=bg, border=border, **s)

if analyse_btn:
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        r = analyse(user_input.strip())

        st.write("")

        # Result card
        st.markdown(f"""
        <div style="background:{r['bg']};border:2px solid {r['border']};border-radius:22px;padding:1.4rem 1.6rem;box-shadow:3px 4px 0px {r['border']}80;margin-bottom:1rem">
            <div style="display:flex;align-items:center;gap:1rem">
                <div style="font-size:3.2rem;line-height:1">{r['emoji']}</div>
                <div>
                    <div style="font-size:2rem;font-weight:900;color:{r['color']};line-height:1;font-family:'Nunito',sans-serif">{r['label']}</div>
                    <div style="font-size:0.78rem;color:#8a7e72;margin-top:0.2rem;font-family:'Nunito',sans-serif">
                        VADER compound:
                        <code style="color:#2a2520;background:rgba(255,255,255,0.7);padding:0.1rem 0.4rem;border-radius:6px;font-family:'DM Mono',monospace">{r['compound']:+.3f}</code>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score bars
        st.markdown('<p style="font-size:0.72rem;font-weight:900;color:#8a7e72;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem">Score Breakdown</p>', unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 4])
        with col_l:
            st.markdown('<span style="font-size:0.82rem;font-weight:800;color:#4e9268">Positive</span>', unsafe_allow_html=True)
        with col_r:
            st.progress(r["pos"])

        col_l, col_r = st.columns([1, 4])
        with col_l:
            st.markdown('<span style="font-size:0.82rem;font-weight:800;color:#c49a2a">Neutral</span>', unsafe_allow_html=True)
        with col_r:
            st.progress(r["neu"])

        col_l, col_r = st.columns([1, 4])
        with col_l:
            st.markdown('<span style="font-size:0.82rem;font-weight:800;color:#c95c5c">Negative</span>', unsafe_allow_html=True)
        with col_r:
            st.progress(r["neg"])

        st.write("")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Positive",  f"{r['pos']:.3f}")
        m2.metric("Negative",  f"{r['neg']:.3f}")
        m3.metric("Neutral",   f"{r['neu']:.3f}")
        m4.metric("Compound",  f"{r['compound']:+.3f}")

        st.write("")

        si_map = {
            "Positive": "😊 People expressing satisfaction engage more with platforms — a key insight in Social Informatics when studying human–technology interaction.",
            "Negative": "💬 Negative feedback signals friction in the user–system relationship — exactly what Social Informatics investigates to improve digital experiences.",
            "Neutral":  "🔍 Neutral language often reflects transactional intent — important when modelling online behaviour in Social Informatics research.",
        }
        st.info(si_map[r['label']])

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

    Label is determined by whichever of **pos / neg / neu** scores highest.
    The **compound score** (−1 to +1) is shown as additional context.
    """)

st.markdown("""
<div style="text-align:center;margin-top:2rem;opacity:0.3">
    <svg width="180" height="12" viewBox="0 0 180 12">
        <path d="M0 6 Q 22 1, 45 6 Q 67 11, 90 6 Q 112 1, 135 6 Q 157 11, 180 6"
              fill="none" stroke="#2a2520" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
</div>
<div style="text-align:center;font-size:0.72rem;color:#b0a898;padding:0.5rem 0 1rem;font-style:italic">Social Informatics Group Assignment · SentiScope · WIX3002</div>
""", unsafe_allow_html=True)
