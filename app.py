from dotenv import load_dotenv
load_dotenv()  # must run before any core/ or utils/ import

import os
import tempfile
import time

import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; max-width: 1150px; }
.app-title { font-size: 2.1rem; font-weight: 700; margin: 0; }
.app-sub { color: #8b8ba7; margin: .2rem 0 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("result", None)
st.session_state.setdefault("chat", [])


# ── Pipeline ─────────────────────────────────────────────────────────────────
def analyse(source: str, language: str):
    """Run the full pipeline with live progress. Returns a result dict or None."""
    with st.status("Starting…", expanded=True) as status:

        def step(label, fn, *args):
            status.update(label=f"{label}…")
            t0 = time.time()
            out = fn(*args)
            st.write(f"✅ {label} ({time.time() - t0:.1f}s)")
            return out

        try:
            chunks = step("Processing audio", process_input, source)
            transcript = step("Transcribing speech", transcribe_all, chunks, language)
            title = step("Generating title", generate_title, transcript)
            summary = step("Writing summary", summarize, transcript)
            actions, decisions, questions = step(
                "Extracting action items, decisions and questions",
                lambda t: (
                    extract_action_items(t),
                    extract_key_decisions(t),
                    extract_questions(t),
                ),
                transcript,
            )
            rag_chain = step("Building chat index", build_rag_chain, transcript)
        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"{type(e).__name__}: {e}")
            return None

        status.update(label="Analysis complete", state="complete", expanded=False)

    return {
        "title": title,
        "language": language,
        "transcript": transcript,
        "summary": summary,
        "action_items": actions,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def build_report(r: dict) -> str:
    return (
        f"# {r['title']}\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action items\n{r['action_items']}\n\n"
        f"## Key decisions\n{r['key_decisions']}\n\n"
        f"## Open questions\n{r['open_questions']}\n"
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎬 AI Video Assistant")

    mode = st.radio("Video source", ["YouTube URL / file path", "Upload a file"])
    source, upload = "", None
    if mode == "Upload a file":
        upload = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mkv", "mov", "webm", "mp3", "wav", "m4a"],
        )
    else:
        source = st.text_input("YouTube URL or local path", placeholder="https://youtu.be/...")

    language = st.selectbox("Spoken language", ["english", "hinglish"])
    run_btn = st.button("Analyse video", type="primary", use_container_width=True)

    r = st.session_state.result
    if r:
        st.divider()
        st.caption("Export")
        st.download_button(
            "Download report (.md)", build_report(r), "meeting_report.md",
            use_container_width=True,
        )
        st.download_button(
            "Download transcript (.txt)", r["transcript"], "transcript.txt",
            use_container_width=True,
        )
        if st.button("Clear results", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat = []
            st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">AI Video Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Turn a video into a transcript, summary and a chat you can ask questions.</p>',
    unsafe_allow_html=True,
)

# ── Run ──────────────────────────────────────────────────────────────────────
if run_btn:
    if upload is not None:
        suffix = os.path.splitext(upload.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(upload.getbuffer())
            source = f.name

    if not source.strip():
        st.warning("Enter a YouTube URL or file path, or upload a file, then click Analyse video.")
    else:
        st.session_state.result = analyse(source.strip(), language)
        st.session_state.chat = []

# ── Results ──────────────────────────────────────────────────────────────────
r = st.session_state.result
if r:
    st.header(r["title"])
    m1, m2 = st.columns(2)
    m1.metric("Transcript length", f"{len(r['transcript'].split()):,} words")
    m2.metric("Language", r["language"].capitalize())

    tab_overview, tab_transcript = st.tabs(["Overview", "Transcript"])

    with tab_overview:
        st.subheader("Summary")
        st.markdown(r["summary"])

        c1, c2, c3 = st.columns(3)
        for col, heading, key in [
            (c1, "✅ Action items", "action_items"),
            (c2, "🔑 Key decisions", "key_decisions"),
            (c3, "❓ Open questions", "open_questions"),
        ]:
            with col, st.container(border=True):
                st.markdown(f"**{heading}**")
                st.markdown(r[key])

    with tab_transcript:
        st.text_area(
            "Transcript", r["transcript"], height=420, label_visibility="collapsed"
        )

    st.divider()
    st.subheader("Ask about this video")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. What were the main decisions?"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript…"):
                try:
                    answer = ask_question(r["rag_chain"], prompt)
                except Exception as e:
                    answer = f"Could not get an answer: {e}"
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

elif not run_btn:
    st.info("Paste a YouTube URL or upload a file in the sidebar, then click **Analyse video**.")