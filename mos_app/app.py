"""Avatar Quality Trial - single-stimulus ACR (Absolute Category Rating) MOS tool for the
AvatarVerse study.

Flow: consent -> demographics -> 3 practice trials -> the participant's assigned session's
trial items -> completion. Each trial embeds a live, free-orbit interactive 3D viewer
(viewer/viewer.html, three.js + OrbitControls + DRACOLoader - see that file and
pipeline/geometry_export.py for how the geometry it loads is built and compressed) and asks
for a single 1-5 ITU-T ACR-style quality rating. Pairwise comparison (this app's earlier design)
is retired for this track - see mos_app/github_store.py's RESPONSES_PATH comment for why
responses land in a new file rather than the old pairwise responses.csv.

Each response is appended, server-side, to a CSV committed back to this repo via the GitHub
Contents API - the token never reaches the participant's browser.

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
import streamlit.components.v1 as components

from github_store import append_response_row

# ---------------------------------------------------------------------- config
# Where the viewer (viewer.html/viewer.js/vendor/) and the geometry bundles it loads are hosted.
# Local testing: both point at a local http.server (see mos_app/README or the Phase 2 test
# notes). Deployment: pushed to a public GitHub-LFS-backed repo, same pattern the earlier
# pairwise track used for video hosting.
VIEWER_BASE = "http://localhost:8791/viewer"
GEOMETRY_BASE = "http://localhost:8791/bundles"
TRIAL_ITEMS_PATH = Path(__file__).parent / "trial_items.csv"

SUBJECTS = ["0000", "0100", "0500", "0450", "0150", "0250"]
CLIPS = ["walk", "front_kick", "bmlmovi_walk", "run", "pickup_box"]
KNOWN_SESSIONS = [f"{s}_{c}" for s in SUBJECTS for c in CLIPS]

# Study-design cap (not just a testing knob): the free-orbit viewer's per-instance download is
# large enough (~45MB, see pipeline/geometry_export.py's module docstring) that a full 37-item
# session was judged too much data (~1.5-1.9GB) - capped to 20 items/session instead, chosen
# after measuring the real bundle sizes against session-length alternatives.
MAIN_TRIALS_LIMIT = 20

# ITU-T ACR-style absolute quality scale - unchanged from the pairwise-era app.
RATING_OPTIONS = ["Select a rating", "1 - Bad", "2 - Poor", "3 - Fair", "4 - Good", "5 - Excellent"]


def _rating_value(label):
    return None if label == RATING_OPTIONS[0] else int(label[0])


PRACTICE_ITEMS = [
    dict(trial_id="practice-1", subject="0000", clip="walk",
        distortion_type="none", severity="none", asset_id="0000_walk_reference"),
    dict(trial_id="practice-2", subject="0000", clip="walk",
        distortion_type="jitter", severity="severe", asset_id="0000_walk_jitter_severe"),
    dict(trial_id="practice-3", subject="0000", clip="walk",
        distortion_type="mesh_decimation", severity="severe", asset_id="0000_walk_mesh_decimation_severe"),
]

st.set_page_config(page_title="Avatar Quality Trial", page_icon="🎬", layout="wide",
                   initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------- data loading
@st.cache_data
def load_trial_items():
    with open(TRIAL_ITEMS_PATH) as f:
        return list(csv.DictReader(f))


def viewer_url(item):
    return f"{VIEWER_BASE}/viewer.html?bundle={GEOMETRY_BASE}/{item['asset_id']}"


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
        all_items = load_trial_items()
        subject, clip = ss.session.split("_", 1)
        session_items = [r for r in all_items if r["subject"] == subject and r["clip"] == clip]
        rng = random.Random(ss.participant_id)
        rng.shuffle(session_items)
        if MAIN_TRIALS_LIMIT is not None:
            session_items = session_items[:MAIN_TRIALS_LIMIT]
        ss.main_items = session_items
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
        '<p class="dim">You\'ll see a 3D avatar in motion, one at a time. Drag to rotate it and '
        'scroll to zoom - look at it from a few angles - then rate its overall quality from 1 '
        '(Bad) to 5 (Excellent). There\'s no right answer &mdash; just go with your first '
        'impression. A few practice trials come first.</p>',
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


def render_trial(item, phase, index, total):
    ss = st.session_state
    if ss.trial_started_at is None:
        ss.trial_started_at = time.time()

    label = "Practice" if phase == "practice" else "Trial"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0 18px;"><span class="eyebrow">{label}</span>'
        f'<span class="dim" style="font-family:monospace;">{index + 1} / {total}</span></div>',
        unsafe_allow_html=True)

    components.iframe(viewer_url(item), height=650, scrolling=False)
    st.markdown(
        '<p class="dim" style="text-align:center;margin-top:8px;">Drag to rotate &middot; scroll to zoom</p>',
        unsafe_allow_html=True)

    rating_label = st.selectbox("Rate this avatar's overall quality", RATING_OPTIONS,
        key=f"rating-{item['trial_id']}")
    rating = _rating_value(rating_label)

    if st.button("Continue", key=f"continue-{item['trial_id']}", use_container_width=True,
                disabled=rating is None):
        record_rating(item, phase, rating)
    if rating is None:
        st.markdown(
            '<p class="dim" style="text-align:center;margin-top:10px;">'
            'Rate the avatar above to continue.</p>', unsafe_allow_html=True)


def record_rating(item, phase, rating):
    ss = st.session_state
    response_ms = int((time.time() - ss.trial_started_at) * 1000)

    if phase == "main":
        append_response_row(dict(
            participant_id=ss.participant_id, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            age=ss.demographics["age"], sex=ss.demographics["sex"], occupation=ss.demographics["occupation"],
            expertise=ss.demographics["expertise"], nationality=ss.demographics["nationality"],
            session=ss.session, trial_id=item["trial_id"], distortion_type=item["distortion_type"],
            severity=item["severity"], asset_id=item["asset_id"], rating=rating, response_ms=response_ms,
        ))

    ss.trial_started_at = None
    if phase == "practice":
        ss.practice_idx += 1
        if ss.practice_idx >= len(PRACTICE_ITEMS):
            ss.stage = "main"
    else:
        ss.main_idx += 1
        if ss.main_idx >= len(ss.main_items):
            ss.stage = "done"
    st.rerun()


def screen_practice():
    render_trial(PRACTICE_ITEMS[st.session_state.practice_idx], "practice",
                st.session_state.practice_idx, len(PRACTICE_ITEMS))


def screen_main():
    ss = st.session_state
    if not ss.main_items:
        st.error("Could not load the trial set for this session. Please refresh to try again.")
        return
    render_trial(ss.main_items[ss.main_idx], "main", ss.main_idx, len(ss.main_items))


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
