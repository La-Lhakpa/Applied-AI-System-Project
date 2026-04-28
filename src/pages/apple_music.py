"""
Apple Music style detail page for a selected recommendation.
"""

import streamlit as st

_PAGE_CSS = """
<style>
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Helvetica Neue", Arial, sans-serif;
    background:
        radial-gradient(circle at 50% 108%, rgba(255, 119, 0, 0.92) 0%, rgba(255, 145, 0, 0.68) 18%, rgba(255, 175, 76, 0.22) 34%, rgba(255, 175, 76, 0.00) 48%),
        linear-gradient(180deg, #b8c4e4 0%, #ceb9d6 40%, #efc0ca 72%, #f6d5c3 100%);
    background-attachment: fixed;
}
.hero-card {
    background: rgba(20, 16, 32, 0.72);
    border-radius: 24px;
    overflow: hidden;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
}
.hero-art {
    width: 100%;
    padding-top: 52%;
    position: relative;
}
.hero-art-inner {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 100px;
}
.hero-body {
    padding: 24px;
}
.hero-title {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.hero-subtitle {
    color: rgba(255,255,255,0.72) !important;
    margin-bottom: 16px;
}
.hero-meta {
    color: rgba(255,255,255,0.88) !important;
    font-size: 14px;
    margin-bottom: 8px;
}
.badge {
    display: inline-block;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    color: #ffffff !important;
    padding: 6px 12px;
    margin-right: 8px;
    margin-top: 8px;
    font-size: 13px;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Apple Music View", page_icon="🎵", layout="centered")
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    selected_song = st.session_state.get("selected_song")
    if not selected_song:
        st.warning("No song selected yet. Open a song from EchoMind recommendations.")
        if st.button("Back to EchoMind", use_container_width=True):
            st.switch_page("app.py")
        return

    st.markdown("## Apple Music")
    st.markdown(
        f"""<div class="hero-card">
            <div class="hero-art" style="background: {selected_song.get("art_gradient", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")};">
                <div class="hero-art-inner">{selected_song.get("emoji", "🎵")}</div>
            </div>
            <div class="hero-body">
                <div class="hero-title">{selected_song.get("title", "Unknown Song")}</div>
                <div class="hero-subtitle">{selected_song.get("artist", "Unknown Artist")}</div>
                <div class="hero-meta">Genre: {selected_song.get("genre", "n/a")} · Mood: {selected_song.get("mood", "n/a")} · Energy: {selected_song.get("energy", 0.0):.2f}</div>
                <span class="badge">Match {selected_song.get("score", 0.0):.2f}</span>
                <span class="badge">Content {selected_song.get("content_score", 0.0):.2f}</span>
                <span class="badge">Label {selected_song.get("label_score", 0.0):.2f}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    play_col, queue_col, back_col = st.columns(3)
    with play_col:
        st.button("Play Now", use_container_width=True)
    with queue_col:
        st.button("Add to Queue", use_container_width=True)
    with back_col:
        if st.button("Back to EchoMind", use_container_width=True):
            st.switch_page("app.py")


if __name__ == "__main__":
    main()
