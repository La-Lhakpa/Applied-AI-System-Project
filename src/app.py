"""
Streamlit UI for TriadTune — Apple Music–inspired minimal layout.

Run with:  streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.pipeline import RecommendationPipeline, RecommendationResult
from src.recommender import Song, UserProfile, load_songs

GENRES = [
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop",
    "country", "classical", "hip hop", "metal", "reggae", "folk", "edm",
    "latin", "soul", "punk",
]

MOODS = [
    "happy", "chill", "intense", "moody", "focused", "relaxed", "uplifting",
    "serene", "playful", "aggressive", "euphoric", "nostalgic", "dreamy",
    "romantic", "melancholic", "dark",
]

_ACCENT = "#FA233B"
_MUTED = "#6E6E73"

_CSS = f"""
<style>
.stApp {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Helvetica Neue", Arial, sans-serif;
}}
h1.triadtune {{
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    font-size: 44px;
}}
p.triadtune-sub {{
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
                


def main() -> None:
    st.set_page_config(page_title="TriadTune", page_icon="🎵", layout="centered")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown('<h1 class="triadtune">TriadTune</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="triadtune-sub">Personalized music recommendations</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Your taste")
        genre = st.selectbox("Genre", GENRES)
        mood = st.selectbox("Mood", MOODS)
        energy = st.slider("Energy", 0.0, 1.0, 0.70, 0.05)
        likes_acoustic = st.checkbox("Prefer acoustic")
        k = st.slider("Show", 3, 10, 5)

        with st.expander("Advanced"):
            blend_alpha = st.slider(
                "Content ↔ Label blend", 0.0, 1.0, 0.5, 0.05,
                help="1.0 → pure audio similarity · 0.0 → pure genre/mood match",
            )

    user = UserProfile(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=likes_acoustic,
    )

    songs = get_songs()
    pipeline = RecommendationPipeline(songs, blend_alpha=blend_alpha)
    results = pipeline.run(user, k=k)

    st.markdown(f"### Top {k} for you")
    for i, r in enumerate(results, 1):
        _render_row(i, r)


if __name__ == "__main__":
    main()
