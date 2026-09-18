"""GitHub-Contents-API-backed CSV storage, shared by app.py (writes) and pages/responses.py
(reads). Credential (GITHUB_TOKEN) stays server-side via st.secrets - never reaches a browser."""
import base64
import csv
import io
from pathlib import Path

import requests
import streamlit as st

RESPONSES_REPO = "shakirul360/AvatarVerse"
RESPONSES_PATH = "mos_app/data/responses.csv"
RESPONSES_FIELDS = [
    "participant_id", "timestamp", "age", "sex", "occupation", "expertise", "nationality",
    "session", "pair_id", "comparison_type", "distortion_type",
    "video_a", "video_b", "chosen_side", "chosen_level", "response_ms",
    "rating_a", "rating_b",   # ITU-T ACR-style 1-5 absolute quality rating, per video
]
LOCAL_FALLBACK = Path(__file__).parent / "local_responses.csv"


def github_headers():
    # st.secrets itself raises StreamlitSecretNotFoundError when no secrets.toml exists at all
    # (not just when a key is missing) - a bare .get() only covers the second case.
    try:
        token = st.secrets.get("GITHUB_TOKEN", None)
    except Exception:
        token = None
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def github_get_file():
    headers = github_headers()
    url = f"https://api.github.com/repos/{RESPONSES_REPO}/contents/{RESPONSES_PATH}"
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    body = r.json()
    content = base64.b64decode(body["content"]).decode("utf-8")
    return content, body["sha"]


def github_put_file(content, sha, message):
    headers = github_headers()
    url = f"https://api.github.com/repos/{RESPONSES_REPO}/contents/{RESPONSES_PATH}"
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload, timeout=15)
    return r.status_code in (200, 201), r


def append_response_row(row: dict):
    """Append one response as a new row of responses.csv on GitHub. Retries a few times on a
    409 (someone else's write landed between our get and put - re-fetch the fresh sha and
    reapply). Falls back to a local file when no GITHUB_TOKEN secret is configured, so the app
    is fully testable with `streamlit run` before deployment."""
    if not github_headers():
        is_new = not LOCAL_FALLBACK.exists()
        with open(LOCAL_FALLBACK, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESPONSES_FIELDS)
            if is_new:
                w.writeheader()
            w.writerow(row)
        return

    for attempt in range(5):
        content, sha = github_get_file()
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=RESPONSES_FIELDS)
        if content is None:
            w.writeheader()
        else:
            buf.write(content)
            if not content.endswith("\n"):
                buf.write("\n")
        w.writerow(row)
        ok, resp = github_put_file(buf.getvalue(), sha, f"Response: {row['participant_id'][:8]}/{row['pair_id']}")
        if ok:
            return
        if resp.status_code == 409:
            import time
            time.sleep(0.3 * (attempt + 1))
            continue
        st.warning(f"Could not save this response to GitHub (status {resp.status_code}). Continuing anyway.")
        return
    st.warning("Could not save this response after several attempts (repeated write conflicts). Continuing anyway.")


def read_all_responses() -> str | None:
    """Current responses.csv content, for the /responses viewer page. Reads live from GitHub
    (not the locally bundled file, which is only as fresh as the last deploy) so it reflects
    every response committed since. Falls back to the local file in the same no-token
    testing mode append_response_row() uses."""
    if not github_headers():
        return LOCAL_FALLBACK.read_text() if LOCAL_FALLBACK.exists() else None
    content, _ = github_get_file()
    return content
