"""
Streamlit interactive UI for TriadTune.

Run with:  streamlit run src/app.py
"""

import streamlit as st

from src.features import FeatureExtractor
from src.pipeline import RecommendationPipeline
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


@st.cache_data
def get_songs() -> list:
    raw = load_songs("data/songs.csv")
    return [Song(**s) for s in raw]


def main() -> None:
    st.set_page_config(
        page_title="TriadTune",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 TriadTune — Personalized Music Recommender")
    st.caption(
        "Simulating how Spotify & TikTok predict what you'll love next  •  "
        "Content-based + Label-based blended pipeline"
    )

    # ── Sidebar: taste profile ────────────────────────────────────────
    with st.sidebar:
        st.header("🎧 Your Taste Profile")
        genre = st.selectbox("Favorite Genre", GENRES)
        mood = st.selectbox("Favorite Mood", MOODS)
        energy = st.slider("Target Energy", 0.0, 1.0, 0.70, 0.05,
                           help="0 = very mellow, 1 = maximum intensity")
        likes_acoustic = st.checkbox("I prefer acoustic songs")
        k = st.slider("Recommendations to show", 3, 10, 5)

        st.divider()
        st.header("⚙️ Pipeline Settings")
        blend_alpha = st.slider(
            "Blend α  (content ←→ label)",
            0.0, 1.0, 0.5, 0.05,
            help="α=1.0 → pure audio-feature cosine similarity\nα=0.0 → pure genre/mood/energy label matching",
        )
        st.caption("α = 1.0 → pure audio features  |  α = 0.0 → pure labels")

    user = UserProfile(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=likes_acoustic,
    )

    songs = get_songs()
    pipeline = RecommendationPipeline(songs, blend_alpha=blend_alpha)
    results = pipeline.run(user, k=k)

    # ── Main layout: recommendations + explainer ──────────────────────
    col_recs, col_edu = st.columns([2, 1])

    with col_recs:
        st.subheader(f"Top {k} Picks for You")
        for i, r in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(
                    f"**{i}. {r.song.title}**  ·  "
                    f"*{r.song.artist}*  ·  "
                    f"`{r.song.genre}` / `{r.song.mood}`"
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Score", f"{r.final_score:.3f}")
                c2.metric("Content (α)", f"{r.content_score:.3f}")
                c3.metric("Label (1-α)", f"{r.label_score:.3f}")
                c4.metric("Energy", f"{r.song.energy:.2f}")
                with st.expander("Why this song?"):
                    st.code(r.explanation, language=None)

    with col_edu:
        st.subheader("📖 How It Works")

        with st.expander("1 · Content-Based Filtering", expanded=True):
            st.markdown("""
**Like Spotify's audio fingerprint model:**

Each song is encoded as a 5-D vector:
```
[energy, valence, danceability, acousticness, tempo_norm]
```
Your preferences are mapped into the **same space** as a *taste vector*.
We score every song with **cosine similarity** — songs whose vector "points in the same direction" as your taste rank highest.
            """)

        with st.expander("2 · Label-Based Filtering"):
            st.markdown("""
**Explicit preference matching:**
```
+1.0  if genre matches
+1.0  if mood matches
+2.0 × energy_similarity   (continuous, not binary)
──────────────────────────
max = 4.0  →  normalized to [0, 1]
```
This simulates early-stage recommendation when a user has just set up their profile and no listening history exists yet.
            """)

        with st.expander("3 · Blended Score"):
            st.markdown(f"""
**final = α · content + (1-α) · label**

Current α = `{blend_alpha:.2f}`

Real platforms blend many more signals:
- Collaborative filtering (*"users like you liked…"*)
- Recency & freshness boosts
- Diversity re-ranking (avoid 5 identical songs)
- Context signals (time of day, activity, device)
            """)

        with st.expander("🧭 Your Taste Vector"):
            extractor = FeatureExtractor()
            taste_vec = extractor.profile_vector(user)
            for name, val in zip(FeatureExtractor.FEATURE_NAMES, taste_vec):
                st.progress(float(val), text=f"`{name}`:  {val:.2f}")


if __name__ == "__main__":
    main()
