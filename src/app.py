"""
Streamlit UI for EchoMind — Apple Music–inspired minimal layout.

Run with:  streamlit run src/app.py
"""

import sys
from html import escape
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.agent_policy import AgentTrace, SessionFeedback
from src.constants import GENRES, MOODS
from src.pipeline import RecommendationPipeline, RecommendationResult
from src.recommender import Song, UserProfile, load_songs

_ACCENT = "#FA233B"
_MUTED = "#6E6E73"
_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "echomind-logo.png"

_CARD_GRADIENTS = [
    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
    "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    "linear-gradient(135deg, #fccb90 0%, #d57eeb 100%)",
    "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)",
    "linear-gradient(135deg, #fd7979 0%, #fecfef 100%)",
    "linear-gradient(135deg, #30cfd0 0%, #667eea 100%)",
]

_MOOD_EMOJIS = {
    "happy": "😊", "chill": "😌", "intense": "⚡", "moody": "🌙",
    "focused": "🎯", "relaxed": "🌿", "energetic": "🔥", "melancholic": "💙",
    "uplifting": "✨", "playful": "🎈", "nostalgic": "🌅", "romantic": "💖",
    "dark": "🌑", "dreamy": "💫", "angry": "🎸", "peaceful": "🕊️",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Helvetica Neue", Arial, sans-serif;
    background:
        radial-gradient(circle at 50% 108%, rgba(255, 119, 0, 0.92) 0%, rgba(255, 145, 0, 0.68) 18%, rgba(255, 175, 76, 0.22) 34%, rgba(255, 175, 76, 0.00) 48%),
        linear-gradient(180deg, #b8c4e4 0%, #ceb9d6 40%, #efc0ca 72%, #f6d5c3 100%);
    background-attachment: fixed;
}
h1.echomind {
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    font-size: 44px;
    text-align: center;
}
p.echomind-sub {
    color: #4a4f5e;
    margin-top: 4px;
    font-size: 15px;
    text-align: center;
}
.username-highlight {
    font-family: 'Great Vibes', cursive;
    font-size: 2em;
    font-weight: 400;
    background: linear-gradient(135deg, #f5576c, #fa709a, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    line-height: 1.1;
}
h3.section-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1b1f2a !important;
    margin: 28px 0 16px;
}
.song-card {
    background: rgba(20, 16, 32, 0.72);
    border-radius: 20px;
    overflow: hidden;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
    margin-bottom: 4px;
}
.card-art {
    width: 100%;
    padding-top: 100%;
    position: relative;
}
.card-art-inner {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 60px;
}
.card-body {
    padding: 14px 16px 16px;
}
.card-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 3px;
}
.card-artist {
    font-size: 13px;
    color: rgba(255,255,255,0.6) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
}
.card-score {
    font-size: 13px;
    color: rgba(255,255,255,0.55) !important;
    font-variant-numeric: tabular-nums;
}
.card-play {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    color: #ffffff !important;
    flex-shrink: 0;
}
/* love / pass button row */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] .stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    font-size: 15px;
    padding: 6px 0;
    cursor: pointer;
}
.stApp .main h1,
.stApp .main h2,
.stApp .main h3,
.stApp .main h4,
.stApp .main p,
.stApp .main div,
.stApp .main span,
.stApp .main li {
    color: #1b1f2a;
}
p.echomind-sub { color: #4a4f5e; }
</style>
"""


@st.cache_data
def get_songs() -> list:
    raw = load_songs("data/songs.csv")
    return [Song(**s) for s in raw]


def _render_recommendations(results: list[RecommendationResult]) -> None:
    if "loved" not in st.session_state:
        st.session_state.loved = set()
    if "passed" not in st.session_state:
        st.session_state.passed = set()

    chunk_size = 3
    for row_start in range(0, len(results), chunk_size):
        row = results[row_start : row_start + chunk_size]
        cols = st.columns(len(row))
        for col, (offset, r) in zip(cols, enumerate(row)):
            rank = row_start + offset + 1
            gradient = _CARD_GRADIENTS[(rank - 1) % len(_CARD_GRADIENTS)]
            emoji = _MOOD_EMOJIS.get(r.song.mood, "🎵")
            song_id = r.song.id
            is_loved = song_id in st.session_state.loved
            is_passed = song_id in st.session_state.passed
            with col:
                st.markdown(
                    f"""<div class="song-card">
                        <div class="card-art" style="background: {gradient};">
                            <div class="card-art-inner">{emoji}</div>
                        </div>
                        <div class="card-body">
                            <div class="card-title">{r.song.title}</div>
                            <div class="card-artist">{r.song.artist}</div>
                            <div class="card-footer">
                                <span class="card-score">{r.final_score:.2f}</span>
                                <div class="card-play">▶</div>
                            </div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                btn_love, btn_pass = st.columns(2)
                with btn_love:
                    love_label = "❤️ Loved" if is_loved else "🤍 Love"
                    if st.button(love_label, key=f"love_{song_id}_{rank}", use_container_width=True):
                        if is_loved:
                            st.session_state.loved.discard(song_id)
                        else:
                            st.session_state.loved.add(song_id)
                            st.session_state.passed.discard(song_id)
                        st.rerun()
                with btn_pass:
                    pass_label = "✗ Passed" if is_passed else "✕ Pass"
                    if st.button(pass_label, key=f"pass_{song_id}_{rank}", use_container_width=True):
                        if is_passed:
                            st.session_state.passed.discard(song_id)
                        else:
                            st.session_state.passed.add(song_id)
                            st.session_state.loved.discard(song_id)
                        st.rerun()
                with st.expander(f"Details · {r.song.genre} · {r.song.mood}"):
                    st.markdown(
                        f"**Content** `{r.content_score:.3f}`  \n"
                        f"**Label** `{r.label_score:.3f}`  \n"
                        f"**Energy** `{r.song.energy:.2f}`"
                    )


def _render_subtitle(username: str) -> None:
    display_name = username.strip()
    if display_name:
        body = (
            f'Personalized music recommendations for '
            f'<span class="username-highlight">{display_name}</span>'
        )
    else:
        body = "Personalized music recommendations"
    st.markdown(f'<p class="echomind-sub">{body}</p>', unsafe_allow_html=True)


def _save_recommendation_snapshot(results: list[RecommendationResult]) -> dict:
    return {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": [
            {
                "title": r.song.title,
                "artist": r.song.artist,
                "genre": r.song.genre,
                "mood": r.song.mood,
                "score": round(r.final_score, 3),
            }
            for r in results
        ],
    }


def _render_saved_recommendations(username: str) -> None:
    snapshots = st.session_state.saved_recommendations.get(username, [])
    if snapshots:
        options = [f"{idx + 1}. {snap['saved_at']}" for idx, snap in enumerate(snapshots)]
        selected = st.selectbox("Load saved set", options)
        selected_idx = options.index(selected)
        with st.expander("Saved recommendations", expanded=False):
            for i, item in enumerate(snapshots[selected_idx]["items"], 1):
                st.markdown(
                    f"{i}. **{item['title']}** — {item['artist']}  \n"
                    f"`{item['genre']}` · `{item['mood']}` · score `{item['score']:.3f}`"
                )
    else:
        st.info("No saved recommendations yet for this username.")


def _render_agent_trace(trace: AgentTrace) -> None:
    policy_step = next((s for s in trace.steps if s.name == "policy_decision"), None)
    guardrail_step = next((s for s in trace.steps if s.name == "guardrail_check"), None)
    finalize_step = next((s for s in trace.steps if s.name == "finalize"), None)

    alpha_display = "baseline"
    confidence_display = "1.00"
    fallback_display = "Yes" if trace.fallback_used else "No"
    returned_display = "n/a"
    if policy_step:
        alpha_value = policy_step.metrics.get("adjusted_blend_alpha")
        if isinstance(alpha_value, (int, float)):
            alpha_display = f"{alpha_value:.2f}"
        confidence_display = f"{policy_step.confidence:.2f}"
    if finalize_step:
        returned_value = finalize_step.metrics.get("returned_count")
        requested_value = finalize_step.metrics.get("requested_k")
        if isinstance(returned_value, int) and isinstance(requested_value, int):
            returned_display = f"{returned_value}/{requested_value}"

    st.markdown("#### Agent summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Policy alpha", alpha_display)
    s2.metric("Policy confidence", confidence_display)
    s3.metric("Returned songs", returned_display)
    s4.metric("Fallback used", fallback_display)
    if guardrail_step and guardrail_step.metrics.get("relaxed_filters"):
        st.info("Guardrail relaxed hard filters to keep enough candidate songs.")

    with st.expander("Agent reasoning trace", expanded=False):
        for idx, step in enumerate(trace.steps, 1):
            st.markdown(
                f"**{idx}. {step.name.replace('_', ' ').title()}**  \n"
                f"{step.decision}  \n"
                f"Confidence: `{step.confidence:.2f}`"
            )
            if step.metrics:
                for key, value in step.metrics.items():
                    st.caption(f"{key}: {value}")
        if trace.final_rationale:
            st.markdown(f"**Final rationale:** {trace.final_rationale}")
        if trace.fallback_used:
            st.warning("Fallback mode was used for this run.")


def main() -> None:
    page_icon = str(_LOGO_PATH) if _LOGO_PATH.exists() else "🎵"
    st.set_page_config(page_title="EchoMind", page_icon=page_icon, layout="centered")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown('<h1 class="echomind">EchoMind</h1>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "saved_recommendations" not in st.session_state:
        st.session_state.saved_recommendations = {}

    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=170)
        username = st.text_input("Username", placeholder="Create temporary profile")
        st.caption("Temporary profile saved for this app session.")
        st.markdown("### Your taste")
        genre = st.selectbox("Genre", GENRES)
        mood = st.selectbox("Mood", MOODS)
        energy = st.slider("Energy", 0.0, 1.0, 0.70, 0.05)
        likes_acoustic = st.checkbox("Prefer acoustic")
        intent_text = st.text_input(
            "Intent (optional)",
            placeholder="e.g., calm coding mix, high-energy workout",
        )
        k = st.slider("Show", 3, 10, 5)

        with st.expander("Advanced"):
            blend_alpha = st.slider(
                "Content ↔ Label blend", 0.0, 1.0, 0.5, 0.05,
                help="1.0 → pure audio similarity · 0.0 → pure genre/mood match",
            )
            likes_count = st.number_input("Session likes", min_value=0, value=0, step=1)
            skips_count = st.number_input("Session skips", min_value=0, value=0, step=1)

    _render_subtitle(username)

    user = UserProfile(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=likes_acoustic,
    )

    songs = get_songs()
    pipeline = RecommendationPipeline(songs, blend_alpha=blend_alpha)
    feedback = SessionFeedback(likes=int(likes_count), skips=int(skips_count))
    results, trace = pipeline.run(
        user,
        k=k,
        intent_text=intent_text,
        session_feedback=feedback,
        return_trace=True,
    )

    st.markdown('<h3 class="section-title">Top Songs for You</h3>', unsafe_allow_html=True)
    _render_recommendations(results)
    _render_agent_trace(trace)

    st.markdown("### Save recommendations")
    if username:
        if st.button("Save current recommendations"):
            user_store = st.session_state.saved_recommendations.setdefault(username, [])
            snapshot = _save_recommendation_snapshot(results)
            user_store.append(snapshot)
            st.success(f"Saved {len(results)} recommendations for {username}.")
        _render_saved_recommendations(username)
    else:
        st.info("Enter a username to save and load recommendations for this session.")


if __name__ == "__main__":
    main()
