import html
import os
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def render_html(markup: str) -> None:
    """Render raw HTML safely (strips indentation so Markdown never treats it as code)."""
    st.markdown(
        " ".join(line.strip() for line in markup.splitlines()),
        unsafe_allow_html=True,
    )


def to_items(value) -> list[str]:
    """Normalise extractor output (string or list) into a clean list of items."""
    if value is None:
        return []
    raw = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).splitlines()
    items = []
    for line in raw:
        clean = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).replace("**", "").strip()
        if clean:
            items.append(clean)
    return items


def to_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {v}" for v in value)
    return str(value or "")


def is_youtube(source: str) -> bool:
    return bool(re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", source or ""))


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
render_html("""
<style>
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap)');

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(124,58,237,0.35), transparent 60%),
        radial-gradient(800px 500px at 95% 0%, rgba(6,182,212,0.25), transparent 60%),
        radial-gradient(700px 500px at 50% 110%, rgba(236,72,153,0.18), transparent 60%),
        #0a0b14;
    color: #e7e9f3;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden;
    padding: 44px 48px; border-radius: 28px;
    background: linear-gradient(135deg, rgba(124,58,237,0.28), rgba(6,182,212,0.14));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    display: flex; align-items: center; justify-content: space-between; gap: 32px;
    margin-bottom: 26px;
}
.hero::after {
    content: ""; position: absolute; width: 380px; height: 380px; right: -120px; top: -160px;
    background: radial-gradient(circle, rgba(236,72,153,0.45), transparent 65%);
    filter: blur(20px); animation: float 9s ease-in-out infinite;
}
@keyframes float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(30px) } }
.badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14); color: #c4b5fd;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 12px #34d399; }
.hero h1 {
    font-size: 46px; line-height: 1.08; font-weight: 800; margin: 16px 0 12px; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 45%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #b6bbd1; font-size: 16.5px; max-width: 560px; margin: 0; line-height: 1.6; }

/* animated waveform graphic */
.wave { display: flex; align-items: center; gap: 7px; height: 150px; z-index: 1; padding-right: 10px; }
.wave span {
    width: 9px; border-radius: 99px;
    background: linear-gradient(180deg, #a78bfa, #22d3ee);
    box-shadow: 0 0 18px rgba(167,139,250,0.55);
    animation: bounce 1.4s ease-in-out infinite;
}
@keyframes bounce { 0%,100% { height: 22px } 50% { height: 130px } }
@media (max-width: 900px) { .wave { display: none } .hero h1 { font-size: 34px } .hero { padding: 30px } }

/* ---------- Cards ---------- */
.feature {
    padding: 24px; border-radius: 22px; height: 100%;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.feature:hover { transform: translateY(-5px); border-color: rgba(167,139,250,0.5); box-shadow: 0 18px 44px rgba(124,58,237,0.22); }
.icon {
    width: 50px; height: 50px; border-radius: 15px; display: flex; align-items: center; justify-content: center;
    font-size: 23px; margin-bottom: 14px;
    background: linear-gradient(135deg, #7c3aed, #06b6d4); box-shadow: 0 10px 26px rgba(124,58,237,0.4);
}
.feature h4 { margin: 0 0 6px; font-size: 16.5px; font-weight: 700; color: #fff; }
.feature p { margin: 0; color: #a3a9c2; font-size: 14px; line-height: 1.55; }

.stat {
    padding: 20px 22px; border-radius: 20px;
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09);
}
.stat .label { font-size: 12px; color: #9aa0ba; text-transform: uppercase; letter-spacing: .09em; font-weight: 600; }
.stat .value {
    font-size: 32px; font-weight: 800; margin-top: 4px;
    background: linear-gradient(90deg, #fff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.title-card {
    padding: 26px 30px; border-radius: 22px; margin: 6px 0 20px;
    background: linear-gradient(120deg, rgba(124,58,237,0.22), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.10);
}
.title-card .kicker { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: #67e8f9; font-weight: 700; }
.title-card h2 { margin: 8px 0 0; font-size: 28px; font-weight: 800; color: #fff; letter-spacing: -0.01em; }

.item {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 15px 18px; margin-bottom: 10px; border-radius: 16px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid var(--accent);
    transition: background .2s ease, transform .2s ease;
}
.item:hover { background: rgba(255,255,255,0.07); transform: translateX(4px); }
.item .num {
    flex: 0 0 28px; height: 28px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
    font-size: 12.5px; font-weight: 700; color: #0a0b14; background: var(--accent);
}
.item .txt { color: #dfe2f0; font-size: 15px; line-height: 1.55; }
.empty { padding: 26px; text-align: center; color: #8d93ad; border: 1px dashed rgba(255,255,255,0.14); border-radius: 16px; }
.section-h { font-size: 18px; font-weight: 700; color: #fff; margin: 6px 0 14px; }

/* ---------- Streamlit widgets ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.035); border-radius: 22px !important;
    border-color: rgba(255,255,255,0.09) !important; backdrop-filter: blur(10px);
}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important; color: #fff !important;
    border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.12) !important;
}
.stTextInput input { padding: 13px 16px !important; font-size: 15px !important; }
.stTextInput input:focus { border-color: #a78bfa !important; box-shadow: 0 0 0 3px rgba(167,139,250,0.22) !important; }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04); border: 1.5px dashed rgba(167,139,250,0.45); border-radius: 18px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 14px; padding: 0.62rem 1.2rem; font-weight: 600;
    background: rgba(255,255,255,0.06); color: #e7e9f3; border: 1px solid rgba(255,255,255,0.13);
    transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #a78bfa; color: #fff; transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #7c3aed, #06b6d4); border: none; color: #fff;
    box-shadow: 0 12px 30px rgba(124,58,237,0.42);
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 16px 40px rgba(6,182,212,0.45); filter: brightness(1.08); }

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; padding: 6px; border-radius: 16px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 11px; padding: 9px 18px; color: #a3a9c2; font-weight: 600; background: transparent;
}
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #7c3aed, #4f46e5); color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 14px 16px;
}
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stStatusWidget"], [data-testid="stExpander"] { border-radius: 18px; }
</style>
""")


# ----------------------------------------------------------------------------
# UI building blocks
# ----------------------------------------------------------------------------
def hero() -> None:
    bars = "".join(
        f'<span style="animation-delay:{i * 0.11:.2f}s"></span>' for i in range(12)
    )
    render_html(f"""
    <div class="hero">
        <div style="z-index:1">
            <span class="badge"><span class="dot"></span> AI Video Assistant</span>
            <h1>Turn any video into<br>clear, actionable insight</h1>
            <p>Paste a YouTube link or upload a recording. Get the transcript, a summary,
            action items, key decisions and open questions, then chat with the content.</p>
        </div>
        <div class="wave">{bars}</div>
    </div>
    """)


def feature_cards() -> None:
    features = [
        ("🎧", "Accurate transcription", "English and Hinglish audio converted into clean, readable text."),
        ("🧠", "Smart summaries", "Title, summary, action items, decisions and open questions in one pass."),
        ("💬", "Chat with your video", "Ask follow-up questions and get answers grounded in the transcript."),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), features):
        with col:
            render_html(f"""
            <div class="feature">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """)


def stat_tiles(result: dict) -> None:
    words = len(result["transcript"].split())
    stats = [
        ("Words transcribed", f"{words:,}"),
        ("Action items", len(to_items(result["action_items"]))),
        ("Key decisions", len(to_items(result["key_decisions"]))),
        ("Open questions", len(to_items(result["open_questions"]))),
    ]
    for col, (label, value) in zip(st.columns(4), stats):
        with col:
            render_html(f"""
            <div class="stat">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """)


def item_list(value, accent: str, empty_msg: str) -> None:
    items = to_items(value)
    if not items:
        render_html(f'<div class="empty">{empty_msg}</div>')
        return
    rows = "".join(
        f'<div class="item" style="--accent:{accent}">'
        f'<div class="num">{i}</div><div class="txt">{html.escape(text)}</div></div>'
        for i, text in enumerate(items, 1)
    )
    render_html(rows)


def build_report(result: dict) -> str:
    return (
        f"# {result['title']}\n\n"
        f"## Summary\n{to_text(result['summary'])}\n\n"
        f"## Action Items\n{to_text(result['action_items'])}\n\n"
        f"## Key Decisions\n{to_text(result['key_decisions'])}\n\n"
        f"## Open Questions\n{to_text(result['open_questions'])}\n\n"
        f"## Transcript\n{result['transcript']}\n"
    )


# ----------------------------------------------------------------------------
# Pipeline with live progress
# ----------------------------------------------------------------------------
def run_pipeline_ui(source: str, language: str) -> dict:
    with st.status("Analyzing your video…", expanded=True) as status:
        st.write("🎧 Preparing audio…")
        chunks = process_input(source)

        st.write("📝 Transcribing speech…")
        transcript = transcribe_all(chunks, language)

        st.write("🏷️ Generating title and summary…")
        title = generate_title(transcript)
        summary = summarize(transcript)

        st.write("🔍 Extracting action items, decisions and questions…")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        st.write("🧩 Building the chat knowledge base…")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Analysis complete", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("source", "")


# ----------------------------------------------------------------------------
# Page: input
# ----------------------------------------------------------------------------
def input_page() -> None:
    hero()

    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            url_tab, file_tab = st.tabs(["🔗  YouTube link", "📁  Upload file"])
            with url_tab:
                url = st.text_input(
                    "YouTube URL",
                    placeholder="[youtube.com](https://www.youtube.com/watch?v=)",
                    label_visibility="collapsed",
                )
            with file_tab:
                uploaded = st.file_uploader(
                    "Upload audio or video",
                    type=["mp4", "mp3", "wav", "m4a", "mkv", "mov", "webm"],
                    label_visibility="collapsed",
                )
        with right:
            language = st.selectbox("Language", ["English", "Hinglish"])
            go = st.button("✨  Analyze", type="primary", use_container_width=True)

    if go:
        tmp_path = None
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            source = tmp_path
        else:
            source = url.strip()

        if not source:
            st.warning("Paste a YouTube link or upload a file to begin.")
        else:
            try:
                result = run_pipeline_ui(source, language.lower())
                st.session_state.result = result
                st.session_state.source = source
                st.session_state.messages = []
                st.rerun()
            except Exception as exc:
                st.error(f"Something went wrong while processing: {exc}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.write("")
    feature_cards()


# ----------------------------------------------------------------------------
# Page: results
# ----------------------------------------------------------------------------
def results_page() -> None:
    result = st.session_state.result

    top_l, top_r = st.columns([5, 1])
    with top_l:
        render_html('<span class="badge"><span class="dot"></span> Analysis ready</span>')
    with top_r:
        if st.button("＋ New analysis", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.session_state.source = ""
            st.rerun()

    render_html(f"""
    <div class="title-card">
        <div class="kicker">Video title</div>
        <h2>{html.escape(str(result['title']))}</h2>
    </div>
    """)

    stat_tiles(result)
    st.write("")

    overview, insights, chat, transcript = st.tabs(
        ["📋  Overview", "✅  Insights", "💬  Chat", "📝  Transcript"]
    )

    # ---- Overview ----
    with overview:
        st.write("")
        source = st.session_state.source
        if is_youtube(source):
            col_a, col_b = st.columns([3, 2])
        else:
            col_a, col_b = st.container(), None

        with col_a:
            with st.container(border=True):
                render_html('<div class="section-h">Summary</div>')
                st.markdown(to_text(result["summary"]))
        if col_b is not None:
            with col_b:
                st.video(source)

        st.write("")
        d1, d2, _ = st.columns([1, 1, 3])
        with d1:
            st.download_button(
                "⬇  Full report (.md)",
                build_report(result),
                file_name="video_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇  Transcript (.txt)",
                result["transcript"],
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ---- Insights ----
    with insights:
        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_html('<div class="section-h">✅ Action items</div>')
            item_list(result["action_items"], "#34d399", "No action items found.")
        with c2:
            render_html('<div class="section-h">🔑 Key decisions</div>')
            item_list(result["key_decisions"], "#a78bfa", "No key decisions found.")
        with c3:
            render_html('<div class="section-h">❓ Open questions</div>')
            item_list(result["open_questions"], "#fbbf24", "No open questions found.")

    # ---- Chat ----
    with chat:
        st.write("")
        history = st.container(height=440)
        with history:
            if not st.session_state.messages:
                render_html(
                    '<div class="empty">Ask anything about this video. '
                    "Answers come from the transcript.</div>"
                )
            for msg in st.session_state.messages:
                avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        suggestions = [
            "What are the main takeaways?",
            "Who is responsible for what?",
            "What should happen next?",
        ]
        picked = None
        for col, text in zip(st.columns(len(suggestions)), suggestions):
            with col:
                if st.button(text, use_container_width=True, key=f"sugg_{text}"):
                    picked = text

        prompt = st.chat_input("Ask a question about this video…") or picked
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with history:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = ask_question(result["rag_chain"], prompt)
                        except Exception as exc:
                            answer = f"Sorry, I couldn't answer that: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": str(answer)})
            st.rerun()

    # ---- Transcript ----
    with transcript:
        st.write("")
        query = st.text_input(
            "Search transcript",
            placeholder="🔎  Search inside the transcript…",
            label_visibility="collapsed",
        )
        text = html.escape(result["transcript"])
        if query.strip():
            pattern = re.compile(re.escape(html.escape(query.strip())), re.IGNORECASE)
            hits = len(pattern.findall(text))
            text = pattern.sub(
                lambda m: f'<mark style="background:#fbbf24;color:#0a0b14;border-radius:4px;padding:0 3px">{m.group(0)}</mark>',
                text,
            )
            st.caption(f"{hits} match(es)")
        with st.container(border=True, height=520):
            st.markdown(
                f'<div style="color:#d5d9ea;line-height:1.8;font-size:15px;white-space:pre-wrap">{text}</div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
if st.session_state.result is None:
    input_page()
else:
    results_page()

