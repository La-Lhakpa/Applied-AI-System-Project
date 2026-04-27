"""
Streamlit UI for EchoMind — Apple Music–inspired minimal layout.

Run with:  streamlit run src/app.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.agent_policy import SessionFeedback
from src.constants import GENRES, MOODS
from src.pipeline import RecommendationPipeline, RecommendationResult
from src.recommender import Song, UserProfile, load_songs

_ACCENT = "#FA233B"
_MUTED = "#6E6E73"
_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "echomind-logo.png"

_CSS = f"""
<style>
.stApp {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Helvetica Neue", Arial, sans-serif;
    background:
        radial-gradient(circle at 50% 108%, rgba(255, 119, 0, 0.92) 0%, rgba(255, 145, 0, 0.68) 18%, rgba(255, 175, 76, 0.22) 34%, rgba(255, 175, 76, 0.00) 48%),
        linear-gradient(180deg, #b8c4e4 0%, #ceb9d6 40%, #efc0ca 72%, #f6d5c3 100%);
    background-attachment: fixed;
}}
h1.echomind {{
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    font-size: 44px;
}}
p.echomind-sub {{
    color: {_MUTED};
    margin-top: 4px;
    font-size: 15px;
}}
.rank-num {{
    color: {_ACCENT};
    font-weight: 700;
    font-size: 28px;
    min-width: 44px;
    text-align: center;
    font-variant-numeric: tabular-nums;
}}
.song-title {{
    font-size: 17px;
    font-weight: 600;
    letter-spacing: -0.01em;
}}
.song-meta {{
    color: {_MUTED};
    font-size: 13px;
    margin-top: 2px;
}}
.row {{
    display: flex;
    align-items: center;
    gap: 18px;
}}
.stApp .main h1,
.stApp .main h2,
.stApp .main h3,
.stApp .main h4,
.stApp .main p,
.stApp .main div,
.stApp .main span,
.stApp .main li {{
    color: #1b1f2a !important;
}}
p.echomind-sub,
.song-meta {{
    color: #2e3445 !important;
}}
@media (prefers-color-scheme: dark) {{
    .stApp .main p,
    .stApp .main label,
    .stApp .main h1,
    .stApp .main h2,
    .stApp .main h3,
    .stApp .main h4,
    .stApp .main h5,
    .stApp .main h6,
    .stApp .main li,
    .stApp .main span,
    .stApp .main div {{
        color: #1b1f2a !important;
    }}
    .stApp .main p.echomind-sub,
    .stApp .main .song-meta {{
        color: #2e3445 !important;
    }}
}}
</style>
"""


@st.cache_data
def get_songs() -> list:
    raw = load_songs("data/songs.csv")
    return [Song(**s) for s in raw]


def _render_row(rank: int, r: RecommendationResult) -> None:
    with st.container(border=True):
        col_info, col_score = st.columns([5, 2])
        with col_info:
            st.markdown(
                f"""<div class="row">
                    <span class="rank-num">{rank}</span>
                    <div>
                        <div class="song-title">{r.song.title}</div>
                        <div class="song-meta">{r.song.artist} · {r.song.genre} · {r.song.mood}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_score:
            with st.expander(f"Score  {r.final_score:.2f}"):
                st.markdown(
                    f"**Content** `{r.content_score:.3f}`  \n"
                    f"**Label** `{r.label_score:.3f}`  \n"
                    f"**Energy** `{r.song.energy:.2f}`"
                )


def _render_subtitle(username: str) -> None:
    display_name = username.strip()
    subtitle = "Personalized music recommendations"
    if display_name:
        subtitle = f"Personalized music recommendations for {display_name}"
    st.markdown(
        f'<p class="echomind-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


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


def main() -> None:
    page_icon = str(_LOGO_PATH) if _LOGO_PATH.exists() else "🎵"
    st.set_page_config(page_title="EchoMind", page_icon=page_icon, layout="centered")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown('<h1 class="echomind">EchoMind</h1>', unsafe_allow_html=True)

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
    results = pipeline.run(user, k=k, intent_text=intent_text, session_feedback=feedback)

    st.markdown(f"### Top {k} for you")
    for i, r in enumerate(results, 1):
        _render_row(i, r)

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
