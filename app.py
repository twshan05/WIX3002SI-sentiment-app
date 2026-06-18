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

/* textarea */
textarea {
    background: var(--card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(74,111,165,0.1) !important; }

/* button */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    letter-spacing: 0.02em;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* hero */
.hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    color: var(--text);
    margin: 0.5rem 0 0.3rem;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.9rem;
    color: var(--muted);
    margin: 0 auto;
    max-width: 420px;
    line-height: 1.6;
}

/* mascot row */
.mascot-row {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin: 1.5rem 0 2rem;
    flex-wrap: wrap;
}
.mascot-card {
    background: var(--card);
    border: 2px solid var(--border);
    border-radius: 20px;
    padding: 1rem 1.2rem;
    text-align: center;
    width: 140px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.mascot-face { font-size: 2.8rem; line-height: 1; margin-bottom: 0.4rem; }
.mascot-name { font-size: 0.78rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.3rem; }
.mascot-name.pos { color: var(--pos); }
.mascot-name.neg { color: var(--neg); }
.mascot-name.neu { color: var(--neu); }
.mascot-quote { font-size: 0.72rem; color: var(--muted); font-style: italic; line-height: 1.4; }

/* input card */
.input-card {
    background: var(--card);
    border: 2px solid var(--border);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.input-label {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

/* example pills */
.pills-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.pill {
    background: var(--bg);
    border: 2px solid var(--border);
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    font-family: 'Nunito', sans-serif;
    font-weight: 700;
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
}
.pill:hover { border-color: var(--accent); color: var(--accent); background: #eef2f8; }

/* result card */
.result-card {
    background: var(--card);
    border: 2px solid var(--border);
    border-radius: 20px;
    padding: 1.6rem;
    margin-top: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    animation: pop 0.3s ease;
}
@keyframes pop { from { opacity:0; transform: scale(0.97); } to { opacity:1; transform: scale(1); } }

.result-top { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.2rem; }
.result-emoji { font-size: 3rem; }
.result-label { font-size: 1.8rem; font-weight: 900; }
.result-label.pos { color: var(--pos); }
.result-label.neg { color: var(--neg); }
.result-label.neu { color: var(--neu); }
.result-compound { font-size: 0.82rem; color: var(--muted); margin-top: 0.1rem; }
.result-compound span { font-family: 'DM Mono', monospace; color: var(--text); font-weight: 600; }

/* bars */
.bar-row { margin-bottom: 0.55rem; }
.bar-meta { display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700; color: var(--muted); margin-bottom: 0.2rem; }
.bar-outer { background: #f0ece6; border-radius: 99px; height: 8px; overflow: hidden; }
.bar-inner { height: 100%; border-radius: 99px; }
.bar-pos { background: var(--pos); }
.bar-neg { background: var(--neg); }
.bar-neu { background: var(--neu); }

/* vader grid */
.vader-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.5rem; margin-top: 1.1rem; padding-top: 1.1rem; border-top: 2px solid var(--border); }
.vader-cell { background: var(--bg); border-radius: 12px; padding: 0.65rem 0.4rem; text-align: center; }
.vader-val { font-size: 1.05rem; font-weight: 800; font-family: 'DM Mono', monospace; color: var(--text); }
.vader-key { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.15rem; font-weight: 700; }

/* si note */
.si-note { background: #eef2f8; border: 2px solid #c8d8f0; border-radius: 12px; padding: 0.7rem 1rem; font-size: 0.82rem; color: #4a6fa5; margin-top: 0.9rem; line-height: 1.5; }

/* footer */
.credits { text-align: center; font-size: 0.72rem; color: #c0b8b0; padding-top: 1.5rem; margin-top: 2rem; border-top: 2px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ── session state for text input ─────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="font-size:0.72rem;letter-spacing:0.15em;text-transform:uppercase;color:#8a8078;margin-bottom:0.4rem;">
        Social Informatics · WIX3002
    </div>
    <div class="hero-title">Senti<span>Scope</span> ✦</div>
    <div class="hero-sub">
        Type a review and discover its emotional tone —
        how language shapes human–technology interaction.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mascots ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mascot-row">
    <div class="mascot-card">
        <div class="mascot-face">🥰</div>
        <div class="mascot-name pos">Positive</div>
        <div class="mascot-quote">"This is absolutely amazing!"</div>
    </div>
    <div class="mascot-card">
        <div class="mascot-face">😐</div>
        <div class="mascot-name neu">Neutral</div>
        <div class="mascot-quote">"It was okay, I guess."</div>
    </div>
    <div class="mascot-card">
        <div class="mascot-face">😤</div>
        <div class="mascot-name neg">Negative</div>
        <div class="mascot-quote">"Terrible — never again!"</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Example pills (Streamlit buttons) ────────────────────────────────────────
st.markdown('<div class="pills-label">Try an example</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😍 Best experience ever!"):
        st.session_state.input_text = "Best experience ever!"
        st.rerun()
with col2:
    if st.button("😐 It was okay, nothing special."):
        st.session_state.input_text = "It was okay, nothing special."
        st.rerun()
with col3:
    if st.button("😤 Terrible service, never again."):
        st.session_state.input_text = "Terrible service, never again."
        st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="Your review",
    value=st.session_state.input_text,
    placeholder="e.g. The food was fantastic and the staff were so friendly!",
    height=110,
    key="review_input",
    label_visibility="visible",
)

analyse_clicked = st.button("Analyse Sentiment →")

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyse(text):
    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)
    c = s["compound"]
    if c >= 0.05:
        return dict(label="Positive", cls="pos", emoji="🥰", **s)
    elif c <= -0.05:
        return dict(label="Negative", cls="neg", emoji="😤", **s)
    else:
        return dict(label="Neutral",  cls="neu", emoji="😐", **s)

if analyse_clicked and user_input.strip():
    r = analyse(user_input.strip())
    pp, np_, nu = int(r["pos"]*100), int(r["neg"]*100), int(r["neu"]*100)

    si_map = {
        "Positive": "People expressing satisfaction tend to engage more with platforms — a key insight in <b>Social Informatics</b> when studying human–technology interaction.",
        "Negative": "Negative feedback signals friction in the user–system relationship — exactly what <b>Social Informatics</b> investigates to improve digital experiences.",
        "Neutral":  "Neutral language often reflects transactional intent — important context when modelling online behaviour in <b>Social Informatics</b> research.",
    }

    st.markdown(f"""
    <div class="result-card">
        <div class="result-top">
            <div class="result-emoji">{r['emoji']}</div>
            <div>
                <div class="result-label {r['cls']}">{r['label']}</div>
                <div class="result-compound">VADER compound: <span>{r['compound']:+.3f}</span></div>
            </div>
        </div>

        <div class="bar-row">
            <div class="bar-meta"><span>Positive</span><span>{pp}%</span></div>
            <div class="bar-outer"><div class="bar-inner bar-pos" style="width:{pp}%"></div></div>
        </div>
        <div class="bar-row">
            <div class="bar-meta"><span>Neutral</span><span>{nu}%</span></div>
            <div class="bar-outer"><div class="bar-inner bar-neu" style="width:{nu}%"></div></div>
        </div>
        <div class="bar-row">
            <div class="bar-meta"><span>Negative</span><span>{np_}%</span></div>
            <div class="bar-outer"><div class="bar-inner bar-neg" style="width:{np_}%"></div></div>
        </div>

        <div class="vader-grid">
            <div class="vader-cell"><div class="vader-val" style="color:var(--pos)">{r['pos']:.3f}</div><div class="vader-key">Positive</div></div>
            <div class="vader-cell"><div class="vader-val" style="color:var(--neg)">{r['neg']:.3f}</div><div class="vader-key">Negative</div></div>
            <div class="vader-cell"><div class="vader-val" style="color:var(--neu)">{r['neu']:.3f}</div><div class="vader-key">Neutral</div></div>
            <div class="vader-cell"><div class="vader-val">{r['compound']:+.3f}</div><div class="vader-key">Compound</div></div>
        </div>

        <div class="si-note">💡 {si_map[r['label']]}</div>
    </div>
    """, unsafe_allow_html=True)

elif analyse_clicked:
    st.warning("Please enter some text first!")

# ── How it works ──────────────────────────────────────────────────────────────
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

st.markdown('<div class="credits">Social Informatics Group Assignment · SentiScope · WIX3002</div>', unsafe_allow_html=True)
