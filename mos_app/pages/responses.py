"""Researcher-only view of collected responses, at <app-url>/responses.

Password-gated (ADMIN_PASSWORD secret) - deliberately not open to anyone with the app link,
since a participant browsing raw response data (including other participants' choices) could
bias their own answers, and it's simply not great practice to expose even-anonymous response
data on an unauthenticated URL. Reads live from GitHub on every load, not a locally bundled
file, so it's never stale relative to what participants have actually submitted.
"""
import csv
import io

import streamlit as st

from github_store import read_all_responses

st.set_page_config(page_title="Responses - Avatar Quality Trial", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .stApp { background: #0b0d10; }
  header[data-testid="stHeader"] { background: #0b0d10; }
  #MainMenu, footer { visibility: hidden; }
  h1, h2, h3, p, label, .stMarkdown { color: #e7e9ec !important; }
  div[data-testid="stMetric"] { background: #15181c; border: 1px solid #262b31;
    border-radius: 8px; padding: 12px 16px; }
</style>
""", unsafe_allow_html=True)


def _check_password():
    try:
        expected = st.secrets.get("ADMIN_PASSWORD", None)
    except Exception:
        expected = None
    if not expected:
        st.error(
            "This page needs an ADMIN_PASSWORD secret configured before it can show anything "
            "- add it alongside GITHUB_TOKEN in your app's Settings -> Secrets.")
        return False
    if st.session_state.get("responses_authed"):
        return True
    pw = st.text_input("Password", type="password")
    if pw and pw == expected:
        st.session_state.responses_authed = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False


st.title("Collected responses")

if not _check_password():
    st.stop()

content = read_all_responses()
if not content:
    st.info("No responses recorded yet.")
    st.stop()

rows = list(csv.DictReader(io.StringIO(content)))

c1, c2, c3 = st.columns(3)
c1.metric("Total responses", len(rows))
c2.metric("Participants", len({r["participant_id"] for r in rows}))
c3.metric("Sessions covered", len({r["session"] for r in rows}))

st.download_button("Download as CSV", content, file_name="responses.csv", mime="text/csv")
st.dataframe(rows, use_container_width=True, hide_index=True)
