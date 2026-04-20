"""
Feature extraction: converts Song objects and UserProfile into normalized numeric vectors.

Each song lives in a 5-dimensional feature space:
  [energy, valence, danceability, acousticness, tempo_norm]

A UserProfile is projected into the same space using mood→valence and
genre→danceability heuristics, mirroring how streaming platforms build
a "taste vector" from explicit preferences before computing similarity.
"""

from typing import List

from src.recommender import Song, UserProfile

# Tempo normalization bounds (BPM range across the catalog)
_TEMPO_MIN = 60.0
_TEMPO_MAX = 180.0

# Approximate valence (positivity) implied by each mood label
_MOOD_VALENCE: dict = {
    "happy": 0.85,
    "uplifting": 0.80,
    "euphoric": 0.90,
    "playful": 0.85,
    "romantic": 0.75,
    "relaxed": 0.65,
    "chill": 0.60,
    "dreamy": 0.60,
    "nostalgic": 0.50,
    "focused": 0.55,
    "serene": 0.55,
    "melancholic": 0.30,
    "moody": 0.35,
    "intense": 0.45,
    "dark": 0.20,
    "aggressive": 0.25,
}

# Approximate danceability implied by each genre
_GENRE_DANCE: dict = {
    "pop": 0.80,
    "edm": 0.85,
    "hip hop": 0.82,
    "latin": 0.85,
    "reggae": 0.75,
    "synthwave": 0.72,
    "indie pop": 0.75,
    "punk": 0.65,
    "rock": 0.60,
    "lofi": 0.58,
    "jazz": 0.55,
    "soul": 0.60,
    "country": 0.55,
    "folk": 0.50,
    "ambient": 0.38,
    "classical": 0.30,
    "metal": 0.45,
}


class FeatureExtractor:
    """
    Converts Song and UserProfile objects into normalized float vectors.

    The same 5-feature space is used for both songs and taste profiles so
    cosine similarity can be computed directly between them.
    """

    FEATURE_NAMES: List[str] = [
        "energy",
        "valence",
        "danceability",
        "acousticness",
        "tempo_norm",
    ]

    def song_vector(self, song: Song) -> List[float]:
        """Return a normalized feature vector for a single song."""
        tempo_norm = (float(song.tempo_bpm) - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)
        return [
            float(song.energy),
            float(song.valence),
            float(song.danceability),
            float(song.acousticness),
            max(0.0, min(1.0, tempo_norm)),
        ]

    def profile_vector(self, user: UserProfile) -> List[float]:
        """
        Project a UserProfile into the same feature space as songs.

        Explicit fields (energy, likes_acoustic) map directly; genre and mood
        are translated through lookup tables to approximate valence and
        danceability — the same heuristic real recommenders use when a user
        has not yet accumulated listening history.
        """
        valence = _MOOD_VALENCE.get(user.favorite_mood.lower().strip(), 0.5)
        dance = _GENRE_DANCE.get(user.favorite_genre.lower().strip(), 0.6)
        acousticness = 0.80 if user.likes_acoustic else 0.15
        # High-energy preference correlates with faster tempos
        tempo_norm = user.target_energy
        return [
            float(user.target_energy),
            valence,
            dance,
            acousticness,
            tempo_norm,
        ]

    def all_song_vectors(self, songs: List[Song]) -> List[List[float]]:
        """Vectorize all songs in the catalog."""
        return [self.song_vector(s) for s in songs]
