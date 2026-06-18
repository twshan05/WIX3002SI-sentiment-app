import streamlit as st
import nltk
import base64
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

st.set_page_config(
    page_title="SentiScope · Social Informatics",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def get_base64_image(image_path):
    path = Path(image_path)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()

bg_img = get_base64_image("sentiscope-bg.png")

positive_face = """
<svg class="doodle-face" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
<circle cx="60" cy="60" r="42" fill="none" stroke="#2f5f9f" stroke-width="5"/>
<circle cx="45" cy="52" r="5" fill="#2f5f9f"/>
<circle cx="75" cy="52" r="5" fill="#2f5f9f"/>
<path d="M40 70 Q60 90 82 70" fill="none" stroke="#2f5f9f" stroke-width="5" stroke-linecap="round"/>
<path d="M35 35 l-8 -8 M85 35 l8 -8" stroke="#c95c5c" stroke-width="4" stroke-linecap="round"/>
</svg>
"""

neutral_face = """
<svg class="doodle-face" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
<circle cx="60" cy="60" r="42" fill="none" stroke="#2f5f9f" stroke-width="5"/>
<circle cx="45" cy="52" r="5" fill="#2f5f9f"/>
<circle cx="75" cy="52" r="5" fill="#2f5f9f"/>
<path d="M43 75 H78" fill="none" stroke="#2f5f9f" stroke-width="5" stroke-linecap="round"/>
</svg>
"""

negative_face = """
<svg class="doodle-face" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
<circle cx="60" cy="60" r="42" fill="none" stroke="#2f5f9f" stroke-width="5"/>
<circle cx="45" cy="58" r="5" fill="#2f5f9f"/>
<circle cx="75" cy="58" r="5" fill="#2f5f9f"/>
<path d="M42 82 Q60 62 80 82" fill="none" stroke="#2f5f9f" stroke-width="5" stroke-linecap="round"/>
<path d="M35 40 L52 48 M85 40 L68 48" stroke="#2f5f9f" stroke-width="5" stroke-linecap="round"/>
</svg>
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;800;900&family=DM+Mono:wght@400&display=swap');

:root {{
    --cream: #f7f3ea;
    --card: rgba(255, 253, 249, 0.88);
    --border: #d4c9b8;
    --blue: #2f5f9f;
    --pos: #4e9268;
    --neg: #c95c5c;
    --neu: #c49a2a;
    --text: #2a2520;
    --muted: #8a7e72;
}}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Nunito', sans-serif !important;
    color: var(--text) !important;
    background-color: var(--cream) !important;
    background-image: url("data:image/png;base64,{bg_img}") !important;
    background-size: cover !important;
    background-position: center top !important;
    background-attachment: fixed !important;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
    display: none !important;
}}

footer {{
    visibility: hidden;
}}

.block-container {{
    padding: 1.2rem 1.5rem 4rem !important;
    max-width: 720px !important;
}}

.hand-card {{
    background: var(--card);
    backdrop-filter: blur(6px);
    border: 2px solid var(--border);
    border-radius: 22px;
    padding: 1rem 0.8rem;
    text-align: center;
    box-shadow: 4px 5px 0px #c4b8a4;
    min-height: 190px;
}}

.doodle-face {{
    width: 82px;
    height: 82px;
    margin: 0.5rem auto 0.4rem auto;
    display: block;
}}

textarea {{
    background: rgba(255, 253, 249, 0.92) !important;
    border: 2px solid var(--border) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    box-shadow: 3px 3px 0px #c4b8a4 !important;
}}

textarea:focus {{
    border-color: var(--blue) !important;
}}

.stButton > button {{
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 50px !important;
    border: 2px solid var(--border) !important;
    background: rgba(255, 253, 249, 0.92) !important;
    color: var(--text) !important;
    box-shadow: 2px 2px 0px #c4b8a4 !important;
    transition: all 0.12s !important;
}}

.stButton > button:hover {{
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    transform: translateY(-1px) !important;
}}

[data-testid="stMetric"] {{
    background: rgba(255, 253, 249, 0.9) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 2px 3px 0px #c4b8a4 !important;
    padding: 0.6rem 0.8rem !important;
}}

[data-testid="stMetricLabel"] p {{
    font-size: 0.65rem !important;
    font-weight: 900 !important;
    color: var(--muted) !important;
}}

[data-testid="stMetricValue"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
}}

[data-testid="stAlert"] {{
    background: rgba(238, 244, 255, 0.9) !important;
    border: 2px solid #b8ccee !important;
    border-radius: 16px !important;
}}

[data-testid="stExpander"] {{
    background: rgba(255, 253, 249, 0.9) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
}}

label[data-testid="stWidgetLabel"] p {{
    font-weight: 900 !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
</style>
""", unsafe_allow_html=True)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.markdown("""
<div style="height:0.5rem"></div>

<h1 style="text-align:center;font-size:3.4rem;font-weight:900;color:#2a2520;
letter-spacing:-0.04em;margin-bottom:0.2rem;font-family:Nunito,sans-serif">
Senti<span style="color:#2f5f9f">Scope</span> ✦
</h1>

<p style="text-align:center;color:#8a7e72;font-style:italic;margin-bottom:1rem">
Type a review · discover its emotional tone
</p>
""", unsafe_allow_html=True)

st.divider()

mc1, mc2, mc3 = st.columns(3)

with mc1:
    st.markdown(f"""
<div class="hand-card">
    <div style="font-size:0.6rem;font-weight:900;color:#4e9268;
    letter-spacing:0.1em;background:#d4edda;border-radius:50px;
    padding:3px 10px;display:inline-block">POSITIVE</div>
    {positive_face}
    <div style="font-size:0.72rem;color:#8a7e72;font-style:italic">
    "This is absolutely amazing!"
    </div>
</div>
""", unsafe_allow_html=True)

with mc2:
    st.markdown(f"""
<div class="hand-card">
    <div style="font-size:0.6rem;font-weight:900;color:#c49a2a;
    letter-spacing:0.1em;background:#fef9e7;border-radius:50px;
    padding:3px 10px;display:inline-block">NEUTRAL</div>
    {neutral_face}
    <div style="font-size:0.72rem;color:#8a7e72;font-style:italic">
    "It was okay, I guess."
    </div>
</div>
""", unsafe_allow_html=True)

with mc3:
    st.markdown(f"""
<div class="hand-card">
    <div style="font-size:0.6rem;font-weight:900;color:#c95c5c;
    letter-spacing:0.1em;background:#fdecea;border-radius:50px;
    padding:3px 10px;display:inline-block">NEGATIVE</div>
    {negative_face}
    <div style="font-size:0.72rem;color:#8a7e72;font-style:italic">
    "Terrible — never again!"
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

st.markdown("""
<p style="font-size:0.72rem;font-weight:900;color:#8a7e72;text-transform:uppercase;
letter-spacing:0.12em;margin-bottom:0.3rem">
✏️ Try an example
</p>
""", unsafe_allow_html=True)

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

user_input = st.text_area(
    "YOUR REVIEW",
    value=st.session_state.input_text,
    placeholder="e.g. The food was fantastic and the staff were so friendly!",
    height=110,
)

analyse_btn = st.button("Analyse Sentiment →", use_container_width=True)

def analyse(text):
    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)

    scores = {
        "Positive": s["pos"],
        "Negative": s["neg"],
        "Neutral": s["neu"],
    }

    label = max(scores, key=scores.get)

    meta = {
        "Positive": (
            positive_face,
            "#4e9268",
            "#fffdf2",
            "#f0cc5f",
            "Your review sounds happy and satisfied."
        ),
        "Negative": (
            negative_face,
            "#c95c5c",
            "#fffdf2",
            "#f0cc5f",
            "Your review sounds frustrated or unhappy."
        ),
        "Neutral": (
            neutral_face,
            "#c49a2a",
            "#fffdf2",
            "#f0cc5f",
            "Your review sounds mostly neutral."
        ),
    }

    face, color, bg, border, message = meta[label]

    return dict(
        label=label,
        face=face,
        color=color,
        bg=bg,
        border=border,
        message=message,
        **s
    )

if analyse_btn:
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        r = analyse(user_input.strip())

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:{r['bg']};border:2px solid {r['border']};
border-radius:24px;padding:1.2rem 1.5rem;
box-shadow:4px 5px 0 {r['border']};margin-bottom:1rem;
display:flex;align-items:center;gap:1rem">

    <div style="width:90px;min-width:90px">
        {r['face']}
    </div>

    <div>
        <div style="font-size:2rem;font-weight:900;color:{r['color']};
        font-family:Nunito,sans-serif">
            {r['label']}
            <code style="font-size:0.8rem;background:rgba(255,255,255,0.75);
            padding:4px 8px;border-radius:8px;color:#2a2520">
            compound {r['compound']:+.3f}
            </code>
        </div>

        <div style="font-size:0.95rem;color:#5f554b;margin-top:0.2rem">
            {r['message']}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<p style="font-size:0.7rem;font-weight:900;color:#8a7e72;text-transform:uppercase;
letter-spacing:0.12em;margin-bottom:0.2rem">
Score Breakdown
</p>
""", unsafe_allow_html=True)

        bl, br = st.columns([1, 4])
        with bl:
            st.markdown("<span style='font-size:0.82rem;font-weight:800;color:#4e9268'>Positive</span>", unsafe_allow_html=True)
        with br:
            st.progress(r["pos"])

        bl, br = st.columns([1, 4])
        with bl:
            st.markdown("<span style='font-size:0.82rem;font-weight:800;color:#c49a2a'>Neutral</span>", unsafe_allow_html=True)
        with br:
            st.progress(r["neu"])

        bl, br = st.columns([1, 4])
        with bl:
            st.markdown("<span style='font-size:0.82rem;font-weight:800;color:#c95c5c'>Negative</span>", unsafe_allow_html=True)
        with br:
            st.progress(r["neg"])

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Positive", f"{r['pos']:.3f}")
        m2.metric("Negative", f"{r['neg']:.3f}")
        m3.metric("Neutral", f"{r['neu']:.3f}")
        m4.metric("Compound", f"{r['compound']:+.3f}")

        si_map = {
            "Positive": "😊 People expressing satisfaction engage more with platforms — a key insight in Social Informatics.",
            "Negative": "💬 Negative feedback signals friction in the user-system relationship.",
            "Neutral": "🔍 Neutral language often reflects transactional intent in online behaviour.",
        }

        st.info(si_map[r["label"]])

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

with st.expander("🔬 How this works"):
    st.markdown("""
This demo uses **VADER** sentiment scoring.

| Step | Method |
|------|--------|
| Feature extraction | VADER sentiment scores |
| Output | Positive / Neutral / Negative |
| Rule | Highest pos / neu / neg score wins |
""")

st.markdown("""
<p style="text-align:center;font-size:0.7rem;color:#b0a898;
font-style:italic;margin-top:2rem">
Social Informatics Group Assignment · SentiScope · WIX3002
</p>
""", unsafe_allow_html=True)