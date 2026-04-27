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

import logging
from dataclasses import dataclass
from typing import List, Optional
from collections.abc import Mapping

from src.features import FeatureExtractor
from src.agent_policy import (
    AgentTrace,
    PolicyDecision,
    SessionFeedback,
    TraceStep,
    decide_policy,
)
from src.recommender import (
    ENERGY_SIMILARITY_WEIGHT,
    GENRE_MATCH_POINTS,
    MOOD_MATCH_POINTS,
    Song,
    UserProfile,
    score_song,
)
from src.similarity import rank_by_similarity

logger = logging.getLogger(__name__)

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

    def run(
        self,
        user: UserProfile,
        k: int = 5,
        intent_text: Optional[str] = None,
        session_feedback: Optional[SessionFeedback | Mapping[str, int]] = None,
        return_trace: bool = False,
    ) -> List[RecommendationResult] | tuple[List[RecommendationResult], AgentTrace]:
        """Score songs, apply optional agentic policy, return top-k results."""
        feedback = self._normalize_feedback(session_feedback)
        trace = AgentTrace()
        trace.steps.append(
            TraceStep(
                name="input_normalization",
                decision="Normalized session feedback into deterministic counters.",
                confidence=1.0,
                metrics={
                    "likes": feedback.likes if feedback else 0,
                    "skips": feedback.skips if feedback else 0,
                },
            )
        )
        policy: Optional[PolicyDecision] = None
        policy_error = ""
        try:
            policy_result = decide_policy(
                user=user,
                default_blend_alpha=self.blend_alpha,
                intent_text=intent_text,
                session_feedback=feedback,
                return_trace=True,
            )
            policy, policy_trace = policy_result
            trace.steps.extend(policy_trace.steps)
            trace.final_rationale = policy_trace.final_rationale
            if policy and policy.is_active:
                logger.info(
                    "Running policy-aware ranking: alpha=%.3f intent_present=%s feedback_present=%s",
                    policy.adjusted_blend_alpha,
                    bool((intent_text or "").strip()),
                    feedback is not None and (feedback.likes + feedback.skips) > 0,
                )
        except Exception as exc:  # deterministic fallback path on policy failure
            policy_error = f"Policy fallback triggered: {exc.__class__.__name__}"
            logger.warning(
                "Policy decision failed; using baseline ranking fallback.",
                exc_info=exc,
            )
            trace.fallback_used = True
            trace.steps.append(
                TraceStep(
                    name="policy_decision",
                    decision="Policy computation failed; using baseline fallback.",
                    confidence=0.0,
                    metrics={"error": exc.__class__.__name__},
                )
            )

        effective_alpha = policy.adjusted_blend_alpha if policy else self.blend_alpha
        label_user = UserProfile(
            favorite_genre=user.favorite_genre,
            favorite_mood=user.favorite_mood,
            target_energy=policy.target_energy if policy else user.target_energy,
            likes_acoustic=user.likes_acoustic,
        )
        taste_vec = self._extractor.profile_vector(user)
        content_scores = rank_by_similarity(taste_vec, self._song_vectors)
        label_scores = self._compute_label_scores(label_user)
        trace.steps.append(
            TraceStep(
                name="candidate_scoring",
                decision="Computed content and label scores for candidate songs.",
                confidence=1.0,
                metrics={
                    "candidate_count": len(self._songs),
                    "effective_alpha": round(effective_alpha, 3),
                    "target_energy": round(label_user.target_energy, 3),
                },
            )
        )

        results: List[RecommendationResult] = []
        dropped_by_filters = 0
        total_policy_adjustment = 0.0
        for i, song in enumerate(self._songs):
            if not self._passes_hard_filters(song, policy):
                dropped_by_filters += 1
                continue
            cs = content_scores[i]
            ls = label_scores[i]
            base_final = effective_alpha * cs + (1.0 - effective_alpha) * ls
            policy_adjust = self._policy_adjustment(song, policy, label_user.target_energy)
            policy_adjust = max(-0.25, min(0.25, policy_adjust))
            total_policy_adjustment += policy_adjust
            final = base_final + policy_adjust
            explanation = self._explain(
                user=user,
                song=song,
                content_score=cs,
                label_score=ls,
                final_score=final,
                blend_alpha=effective_alpha,
                taste_vec=taste_vec,
                song_vec=self._song_vectors[i],
                policy=policy,
                policy_adjustment=policy_adjust,
                policy_error=policy_error,
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

        trace.steps.append(
            TraceStep(
                name="policy_application",
                decision="Applied hard filters and bounded policy score adjustments.",
                confidence=policy.confidence if policy else 1.0,
                metrics={
                    "dropped_by_filters": dropped_by_filters,
                    "remaining_candidates": len(results),
                    "avg_policy_adjustment": round(
                        (total_policy_adjustment / len(results)) if results else 0.0,
                        4,
                    ),
                },
            )
        )

        guardrail_decision = "No guardrail intervention required."
        guardrail_metrics: dict[str, float | int | str | bool] = {
            "required_k": k,
            "candidate_count": len(results),
            "relaxed_filters": False,
        }
        if policy and len(results) < k and (policy.hard_mood_filters or policy.hard_genre_filters):
            relaxed_moods = len(policy.hard_mood_filters)
            relaxed_genres = len(policy.hard_genre_filters)
            policy.hard_mood_filters = set()
            policy.hard_genre_filters = set()
            results = []
            total_policy_adjustment = 0.0
            for i, song in enumerate(self._songs):
                cs = content_scores[i]
                ls = label_scores[i]
                base_final = effective_alpha * cs + (1.0 - effective_alpha) * ls
                policy_adjust = self._policy_adjustment(song, policy, label_user.target_energy)
                policy_adjust = max(-0.25, min(0.25, policy_adjust))
                total_policy_adjustment += policy_adjust
                final = base_final + policy_adjust
                explanation = self._explain(
                    user=user,
                    song=song,
                    content_score=cs,
                    label_score=ls,
                    final_score=final,
                    blend_alpha=effective_alpha,
                    taste_vec=taste_vec,
                    song_vec=self._song_vectors[i],
                    policy=policy,
                    policy_adjustment=policy_adjust,
                    policy_error=policy_error,
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
            policy.fallback_mode = "filter_relaxation"
            trace.fallback_used = True
            guardrail_decision = "Relaxed hard filters to ensure enough candidates."
            guardrail_metrics = {
                "required_k": k,
                "candidate_count": len(results),
                "relaxed_filters": True,
                "relaxed_mood_filters": relaxed_moods,
                "relaxed_genre_filters": relaxed_genres,
            }

        trace.steps.append(
            TraceStep(
                name="guardrail_check",
                decision=guardrail_decision,
                confidence=1.0,
                metrics=guardrail_metrics,
            )
        )

        results.sort(key=lambda r: r.final_score, reverse=True)
        trace.steps.append(
            TraceStep(
                name="finalize",
                decision="Sorted candidates by final score and selected top-k recommendations.",
                confidence=1.0,
                metrics={"returned_count": len(results[:k]), "requested_k": k},
            )
        )
        if not trace.final_rationale:
            trace.final_rationale = policy.rationale if policy else "Baseline ranking retained."
        logger.info(
            "Recommendation run complete: returned=%d requested_k=%d policy_active=%s",
            len(results[:k]),
            k,
            bool(policy and policy.is_active),
        )
        top_results = results[:k]
        return (top_results, trace) if return_trace else top_results

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

    @staticmethod
    def _normalize_feedback(
        session_feedback: Optional[SessionFeedback | Mapping[str, int]],
    ) -> Optional[SessionFeedback]:
        if session_feedback is None:
            return None
        if isinstance(session_feedback, SessionFeedback):
            return session_feedback
        likes = int(session_feedback.get("likes", 0))
        skips = int(session_feedback.get("skips", 0))
        return SessionFeedback(likes=likes, skips=skips)

    @staticmethod
    def _passes_hard_filters(song: Song, policy: Optional[PolicyDecision]) -> bool:
        if not policy:
            return True
        if policy.hard_genre_filters and song.genre.casefold() not in policy.hard_genre_filters:
            return False
        if policy.hard_mood_filters and song.mood.casefold() not in policy.hard_mood_filters:
            return False
        return True

    @staticmethod
    def _policy_adjustment(song: Song, policy: Optional[PolicyDecision], target_energy: float) -> float:
        if not policy:
            return 0.0
        adjustment = 0.0
        adjustment += policy.genre_boosts.get(song.genre.casefold(), 0.0)
        adjustment += policy.mood_boosts.get(song.mood.casefold(), 0.0)
        if policy.acoustic_boost and song.acousticness > 0.6:
            adjustment += policy.acoustic_boost
        if policy.energy_proximity_weight > 0:
            energy_similarity = max(0.0, 1.0 - abs(song.energy - target_energy))
            adjustment += policy.energy_proximity_weight * energy_similarity
        return adjustment

    def _explain(
        self,
        user: UserProfile,
        song: Song,
        content_score: float,
        label_score: float,
        final_score: float,
        blend_alpha: float,
        taste_vec: List[float],
        song_vec: List[float],
        policy: Optional[PolicyDecision],
        policy_adjustment: float,
        policy_error: str,
    ) -> str:
        lines: List[str] = []
        lines.append(
            f"Content similarity (cosine):    {content_score:.3f}"
        )
        lines.append(
            f"Label match score (normalized): {label_score:.3f}"
        )
        lines.append(
            f"Blended base score (α={blend_alpha:.2f}):      "
            f"{blend_alpha * content_score + (1 - blend_alpha) * label_score:.3f}"
        )
        if policy_error:
            lines.append(policy_error)
        if policy:
            lines.append(
                f"Agentic policy adjustment:      {policy_adjustment:+.3f}"
            )
            lines.append(
                f"Agentic policy active:          {policy.is_active}"
            )
            lines.append(f"Policy rationale: {policy.rationale}")
        lines.append(f"Final score after policy:       {final_score:.3f}")
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
