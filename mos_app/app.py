"""Avatar Quality Trial - pairwise-comparison MOS tool for the AvatarVerse study.

Flow: consent -> demographics -> 3 practice trials -> the participant's assigned session's
57 trial pairs -> completion. Videos are served from the public avatarverse-mos-videos repo
(GitHub LFS); each response is appended, server-side, to responses.csv committed back to this
repo via the GitHub Contents API - the token never reaches the participant's browser.

Run locally:   streamlit run app.py
Deploy: push this repo to GitHub, deploy on share.streamlit.io pointing at mos_app/app.py,
       then set the GITHUB_TOKEN secret (see .streamlit/secrets.toml.example).
"""
import csv
import random
import time
import uuid
from pathlib import Path

import streamlit as st

from github_store import append_response_row

# ---------------------------------------------------------------------- config
VIDEO_BASE = "https://media.githubusercontent.com/media/shakirul360/avatarverse-mos-videos/main/"
TRIAL_PAIRS_PATH = Path(__file__).parent / "trial_pairs.csv"

SUBJECTS = ["0000", "0100", "0500", "0450", "0150", "0250"]
CLIPS = ["walk", "front_kick", "bmlmovi_walk", "run", "pickup_box"]
KNOWN_SESSIONS = [f"{s}_{c}" for s in SUBJECTS for c in CLIPS]

# Testing-phase knob: caps each session to its first N (already-shuffled, so effectively
# random) main trials instead of the full 57, for quick end-to-end runs. Set to None to run
# the real study at full length - remember to flip this back before real data collection.
MAIN_TRIALS_LIMIT = 10

PRACTICE_PAIRS = [
    dict(pair_id="practice-1", comparison_type="practice", distortion_type="jitter",
        video_a="single/0000_walk_reference.mp4", video_b="single/0000_walk_jitter_severe.mp4",
        level_a="reference", level_b="severe"),
    dict(pair_id="practice-2", comparison_type="practice", distortion_type="texture_compression",
        video_a="single/0000_walk_reference.mp4", video_b="single/0000_walk_texture_compression_mild.mp4",
        level_a="reference", level_b="mild"),
    dict(pair_id="practice-3", comparison_type="practice", distortion_type="vertex_quantization",
        video_a="single/0000_walk_vertex_quantization_mild.mp4", video_b="single/0000_walk_vertex_quantization_severe.mp4",
        level_a="mild", level_b="severe"),
]

st.set_page_config(page_title="Avatar Quality Trial", page_icon="🎬", layout="wide",
                   initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------- data loading
@st.cache_data
def load_trial_pairs():
    with open(TRIAL_PAIRS_PATH) as f:
        return list(csv.DictReader(f))


def pick_session():
    params = st.query_params
    requested = params.get("session")
    if requested in KNOWN_SESSIONS:
        return requested
    return random.choice(KNOWN_SESSIONS)


# ---------------------------------------------------------------------- session state
def init_state():
    ss = st.session_state
    if "stage" not in ss:
        ss.stage = "welcome"
        ss.participant_id = str(uuid.uuid4())
        ss.session = pick_session()
        ss.demographics = {}
        ss.practice_idx = 0
        ss.main_idx = 0
        all_pairs = load_trial_pairs()
        subject, clip = ss.session.split("_", 1)
        session_pairs = [r for r in all_pairs if r["subject"] == subject and r["clip"] == clip]
        rng = random.Random(ss.participant_id)
        rng.shuffle(session_pairs)
        if MAIN_TRIALS_LIMIT is not None:
            session_pairs = session_pairs[:MAIN_TRIALS_LIMIT]
        ss.main_pairs = session_pairs
        ss.trial_started_at = None


# ---------------------------------------------------------------------- styling
st.markdown("""
<style>
  .stApp { background: #0b0d10; }
  header[data-testid="stHeader"] { background: #0b0d10; }
  #MainMenu, footer, header [data-testid="stToolbarActions"] { visibility: hidden; }
  .block-container { padding-top: 2.5rem; max-width: 1100px; }
  h1, h2, h3, p, label, .stMarkdown { color: #e7e9ec !important; }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #15181c; border: 1px solid #262b31; border-radius: 10px;
  }
  .stButton button {
    background: #4fd8c4; color: #06201c; font-weight: 600; border: none; border-radius: 7px;
  }
  .stButton button:hover { background: #6fe3d3; color: #06201c; }
  .eyebrow { font-family: monospace; font-size: 12px; letter-spacing: 0.08em;
    text-transform: uppercase; color: #4fd8c4; }
  .dim { color: #8b93a1 !important; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------- screens
def screen_welcome():
    st.markdown('<p class="eyebrow">AvatarVerse &middot; Perceptual Quality Study</p>', unsafe_allow_html=True)
    st.title("Avatar Quality Trial")
    st.markdown(
        '<p class="dim">You\'ll see short pairs of videos of a 3D avatar in motion, side by '
        'side. For each pair, choose the one that looks better to you. There\'s no right '
        'answer &mdash; just go with your first impression. A few practice pairs come first.</p>',
        unsafe_allow_html=True)
    consent = st.checkbox(
        "I'm 18 or older and agree to take part in this study. My responses are anonymous "
        "and used only for research analysis of 3D avatar rendering quality.")
    if st.button("Continue", disabled=not consent):
        st.session_state.stage = "demographics"
        st.rerun()


def screen_demographics():
    st.markdown('<p class="eyebrow">Before you start</p>', unsafe_allow_html=True)
    st.title("A few quick questions")
    st.markdown('<p class="dim">Used only for demographic analysis of the study results.</p>', unsafe_allow_html=True)
    with st.form("demographics_form"):
        age = st.radio("What is your age?",
            ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 or above", "Prefer not to say"],
            index=None)
        sex = st.radio("What is your sex?", ["Female", "Male", "Other", "Prefer not to say"], index=None)
        occupation = st.radio("What is your occupation?",
            ["Student", "Researcher / Academic staff", "Engineer / Developer", "Designer / Artist",
             "Industry professional", "Other", "Prefer not to say"], index=None)
        expertise = st.radio(
            "What is your level of expertise in 3D representation, visual quality assessment, or 3D processing?",
            ["Expert", "Advanced", "Intermediate", "Basic", "Not an expert"], index=None)
        nationality = st.selectbox("What is your nationality or region?",
            ["", "Africa", "East Asia", "South Asia", "Southeast Asia", "Middle East", "Europe",
             "North America", "South America", "Oceania", "Prefer not to say"])
        submitted = st.form_submit_button("Start practice trials")
        if submitted:
            if not all([age, sex, occupation, expertise, nationality]):
                st.error("Please answer every question before continuing.")
            else:
                st.session_state.demographics = dict(
                    age=age, sex=sex, occupation=occupation, expertise=expertise, nationality=nationality)
                st.session_state.stage = "practice"
                st.rerun()


def render_trial(pair, phase, index, total):
    ss = st.session_state
    if ss.trial_started_at is None:
        ss.trial_started_at = time.time()

    label = "Practice" if phase == "practice" else "Trial"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0 18px;"><span class="eyebrow">{label}</span>'
        f'<span class="dim" style="font-family:monospace;">{index + 1} / {total}</span></div>',
        unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.video(VIDEO_BASE + pair["video_a"], loop=True, autoplay=True, muted=True)
        if st.button("This one looks better  ←", key=f"choose-a-{pair['pair_id']}", use_container_width=True):
            record_choice(pair, phase, "a")
    with col_b:
        st.video(VIDEO_BASE + pair["video_b"], loop=True, autoplay=True, muted=True)
        if st.button("→  This one looks better", key=f"choose-b-{pair['pair_id']}", use_container_width=True):
            record_choice(pair, phase, "b")


def record_choice(pair, phase, side):
    ss = st.session_state
    response_ms = int((time.time() - ss.trial_started_at) * 1000)

    if phase == "main":
        chosen_level = pair["level_a"] if side == "a" else pair["level_b"]
        append_response_row(dict(
            participant_id=ss.participant_id, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            age=ss.demographics["age"], sex=ss.demographics["sex"], occupation=ss.demographics["occupation"],
            expertise=ss.demographics["expertise"], nationality=ss.demographics["nationality"],
            session=ss.session, pair_id=pair["pair_id"], comparison_type=pair["comparison_type"],
            distortion_type=pair["distortion_type"], video_a=pair["video_a"], video_b=pair["video_b"],
            chosen_side=side, chosen_level=chosen_level, response_ms=response_ms,
        ))

    ss.trial_started_at = None
    if phase == "practice":
        ss.practice_idx += 1
        if ss.practice_idx >= len(PRACTICE_PAIRS):
            ss.stage = "main"
    else:
        ss.main_idx += 1
        if ss.main_idx >= len(ss.main_pairs):
            ss.stage = "done"
    st.rerun()


def screen_practice():
    render_trial(PRACTICE_PAIRS[st.session_state.practice_idx], "practice",
                st.session_state.practice_idx, len(PRACTICE_PAIRS))


def screen_main():
    ss = st.session_state
    if not ss.main_pairs:
        st.error("Could not load the trial set for this session. Please refresh to try again.")
        return
    render_trial(ss.main_pairs[ss.main_idx], "main", ss.main_idx, len(ss.main_pairs))


def screen_done():
    st.markdown('<p class="eyebrow">All done</p>', unsafe_allow_html=True)
    st.title("Thank you for taking part")
    code = st.session_state.participant_id[:8].upper()
    st.markdown(
        f'<p class="dim">Your responses have been recorded. Your completion code is '
        f'<span style="color:#4fd8c4;font-family:monospace;">{code}</span> &mdash; if you were '
        f'sent here as part of a study, keep this for your reference.</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------- main
init_state()
stage = st.session_state.stage
if stage == "welcome":
    screen_welcome()
elif stage == "demographics":
    screen_demographics()
elif stage == "practice":
    screen_practice()
elif stage == "main":
    screen_main()
elif stage == "done":
    screen_done()
