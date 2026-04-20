"""
Recommendation pipeline: orchestrates feature extraction, similarity scoring,
label matching, and blended ranking in a single composable call.

Architecture mirrors how real streaming platforms combine signals:

  ┌─────────────────────────────────────────────────────────────┐
  │                  RecommendationPipeline.run()               │
  │                                                             │
  │  UserProfile  ──► FeatureExtractor ──► taste_vector        │
  │                                            │               │
  │  Song catalog ──► FeatureExtractor ──► song_vectors        │
  │                                            │               │
  │                         cosine_similarity(taste, song) ──► content_score  │
  │                         label_score (genre+mood+energy)                   │
  │                                            │               │
  │              final = α·content + (1-α)·label               │
  │                                            │               │
  │                         sort ↓  →  top-k  +  explanation   │
  └─────────────────────────────────────────────────────────────┘

Tuning blend_alpha:
  α = 1.0  →  pure audio-feature similarity (like Spotify's "audio DNA")
  α = 0.0  →  pure label matching (explicit genre/mood/energy preferences)
  α = 0.5  →  balanced blend (default, good starting point)
"""

from dataclasses import dataclass
from typing import List

from src.features import FeatureExtractor
from src.recommender import (
    ENERGY_SIMILARITY_WEIGHT,
    GENRE_MATCH_POINTS,
    MOOD_MATCH_POINTS,
    Song,
    UserProfile,
    score_song,
)
from src.similarity import rank_by_similarity

# Maximum possible label score (genre + mood + energy_weight * 1.0)
_MAX_LABEL_SCORE = GENRE_MATCH_POINTS + MOOD_MATCH_POINTS + ENERGY_SIMILARITY_WEIGHT


@dataclass
class RecommendationResult:
    """One ranked recommendation with decomposed scores and a human-readable explanation."""

    song: Song
    content_score: float   # cosine similarity in audio-feature space (0–1)
    label_score: float     # normalized genre/mood/energy label match (0–1)
    final_score: float     # blended score used for ranking
    explanation: str


class RecommendationPipeline:
    """
    Modular two-signal recommendation pipeline.

    Parameters
    ----------
    songs:        Full song catalog as Song dataclass instances.
    blend_alpha:  Weight for content signal vs label signal (0.0–1.0).
    """

    def __init__(self, songs: List[Song], blend_alpha: float = 0.5):
        self._songs = songs
        self._extractor = FeatureExtractor()
        self._song_vectors = self._extractor.all_song_vectors(songs)
        self.blend_alpha = blend_alpha

    def run(self, user: UserProfile, k: int = 5) -> List[RecommendationResult]:
        """Score every song, blend signals, return top-k sorted by final_score."""
        taste_vec = self._extractor.profile_vector(user)
        content_scores = rank_by_similarity(taste_vec, self._song_vectors)
        label_scores = self._compute_label_scores(user)

        results: List[RecommendationResult] = []
        for i, song in enumerate(self._songs):
            cs = content_scores[i]
            ls = label_scores[i]
            final = self.blend_alpha * cs + (1.0 - self.blend_alpha) * ls
            explanation = self._explain(
                user, song, cs, ls, taste_vec, self._song_vectors[i]
            )
            results.append(
                RecommendationResult(
                    song=song,
                    content_score=cs,
                    label_score=ls,
                    final_score=final,
                    explanation=explanation,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_label_scores(self, user: UserProfile) -> List[float]:
        scores: List[float] = []
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
        }
        for song in self._songs:
            song_dict = {
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
            }
            raw, _ = score_song(prefs, song_dict)
            scores.append(raw / _MAX_LABEL_SCORE)
        return scores

    def _explain(
        self,
        user: UserProfile,
        song: Song,
        content_score: float,
        label_score: float,
        taste_vec: List[float],
        song_vec: List[float],
    ) -> str:
        lines: List[str] = []
        lines.append(
            f"Content similarity (cosine):    {content_score:.3f}"
        )
        lines.append(
            f"Label match score (normalized): {label_score:.3f}"
        )
        lines.append(
            f"Blended final score (α={self.blend_alpha:.1f}):     "
            f"{self.blend_alpha * content_score + (1 - self.blend_alpha) * label_score:.3f}"
        )
        lines.append("Feature breakdown (song vs your taste):")
        for name, sv, tv in zip(FeatureExtractor.FEATURE_NAMES, song_vec, taste_vec):
            diff = sv - tv
            direction = (
                "↑ above taste" if diff > 0.05
                else ("↓ below taste" if diff < -0.05 else "≈ matches taste")
            )
            lines.append(f"  {name:<14} song={sv:.2f}  taste={tv:.2f}  {direction}")
        if song.genre.casefold() == user.favorite_genre.casefold():
            lines.append(f"Genre: exact match ({song.genre})")
        if song.mood.casefold() == user.favorite_mood.casefold():
            lines.append(f"Mood: exact match ({song.mood})")
        if user.likes_acoustic and song.acousticness > 0.6:
            lines.append(
                f"Acoustic preference satisfied (acousticness={song.acousticness:.2f})"
            )
        return "\n".join(lines)
