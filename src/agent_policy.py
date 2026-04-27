"""
Agentic ranking policy for intent-aware recommendation adjustments.

This module is intentionally lightweight and deterministic:
- It infers a ranking policy from user profile + optional intent text.
- It optionally incorporates simple session feedback (likes/skips).
- It returns transparent policy decisions that can be applied in-pipeline.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, Optional, Set

from src.constants import GENRES, MOODS
from src.recommender import UserProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionFeedback:
    """Lightweight per-session interaction summary."""

    likes: int = 0
    skips: int = 0


@dataclass
class PolicyDecision:
    """Policy outputs consumed by the recommendation pipeline."""

    adjusted_blend_alpha: float
    target_energy: float
    hard_genre_filters: Set[str] = field(default_factory=set)
    hard_mood_filters: Set[str] = field(default_factory=set)
    genre_boosts: Dict[str, float] = field(default_factory=dict)
    mood_boosts: Dict[str, float] = field(default_factory=dict)
    acoustic_boost: float = 0.0
    energy_proximity_weight: float = 0.0
    rationale: str = "No policy adjustments."
    is_active: bool = False


_GENRES = {genre.casefold() for genre in GENRES}
_MOODS = {mood.casefold() for mood in MOODS}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def decide_policy(
    user: UserProfile,
    default_blend_alpha: float,
    intent_text: Optional[str] = None,
    session_feedback: Optional[SessionFeedback] = None,
) -> PolicyDecision:
    """
    Create a deterministic ranking policy from intent + session context.

    Fallback behavior:
    - If no intent and no meaningful feedback, return inactive/no-op policy.
    """
    base_alpha = _clamp(default_blend_alpha, 0.0, 1.0)
    text = (intent_text or "").strip().casefold()
    likes = max(0, session_feedback.likes) if session_feedback else 0
    skips = max(0, session_feedback.skips) if session_feedback else 0

    has_intent = bool(text)
    has_feedback = (likes + skips) > 0
    if not has_intent and not has_feedback:
        logger.info("Agentic policy inactive: no intent text and no session feedback.")
        return PolicyDecision(
            adjusted_blend_alpha=base_alpha,
            target_energy=user.target_energy,
            rationale="No intent/session signal; baseline ranking retained.",
            is_active=False,
        )

    words = set(text.replace("-", " ").split())
    rationale_bits = []
    active = False

    alpha = base_alpha
    target_energy = user.target_energy
    genre_boosts: Dict[str, float] = {}
    mood_boosts: Dict[str, float] = {}
    hard_genre_filters: Set[str] = set()
    hard_mood_filters: Set[str] = set()
    acoustic_boost = 0.0
    energy_proximity_weight = 0.0

    if has_intent:
        for genre in _GENRES:
            genre_tokens = genre.split()
            if all(token in words for token in genre_tokens):
                genre_boosts[genre] = 0.12
                rationale_bits.append(f"Intent mentions genre '{genre}' -> genre boost.")
                active = True
                if "only" in words:
                    hard_genre_filters.add(genre)
                    rationale_bits.append(f"'only' detected -> hard genre filter '{genre}'.")

        for mood in _MOODS:
            if mood in words:
                mood_boosts[mood] = 0.10
                rationale_bits.append(f"Intent mentions mood '{mood}' -> mood boost.")
                active = True
                if "only" in words:
                    hard_mood_filters.add(mood)
                    rationale_bits.append(f"'only' detected -> hard mood filter '{mood}'.")

        if words.intersection({"calm", "chill", "focus", "focused", "study", "relax", "relaxed"}):
            target_energy = _clamp(target_energy - 0.15, 0.0, 1.0)
            alpha = _clamp(alpha - 0.10, 0.0, 1.0)
            energy_proximity_weight = max(energy_proximity_weight, 0.10)
            rationale_bits.append("Calm/focus intent -> lower target energy, favor label alignment.")
            active = True
        if words.intersection({"workout", "gym", "hype", "high", "intense", "run", "running"}):
            target_energy = _clamp(target_energy + 0.15, 0.0, 1.0)
            alpha = _clamp(alpha + 0.10, 0.0, 1.0)
            energy_proximity_weight = max(energy_proximity_weight, 0.10)
            rationale_bits.append("High-energy intent -> higher target energy, favor content signal.")
            active = True
        if words.intersection({"acoustic", "unplugged"}):
            acoustic_boost = 0.08
            rationale_bits.append("Acoustic intent -> boost high-acousticness tracks.")
            active = True

    if has_feedback:
        total = likes + skips
        polarity = (likes - skips) / total
        alpha = _clamp(alpha + 0.08 * polarity, 0.0, 1.0)
        rationale_bits.append(
            f"Session feedback (likes={likes}, skips={skips}) adjusts blend_alpha by {0.08 * polarity:+.3f}."
        )
        active = True

    decision = PolicyDecision(
        adjusted_blend_alpha=alpha,
        target_energy=target_energy,
        hard_genre_filters=hard_genre_filters,
        hard_mood_filters=hard_mood_filters,
        genre_boosts=genre_boosts,
        mood_boosts=mood_boosts,
        acoustic_boost=acoustic_boost,
        energy_proximity_weight=energy_proximity_weight,
        rationale=" ".join(rationale_bits) if rationale_bits else "Policy computed with no-op adjustments.",
        is_active=active,
    )
    logger.info(
        "Agentic policy computed: active=%s alpha=%.3f target_energy=%.3f "
        "hard_genres=%d hard_moods=%d genre_boosts=%d mood_boosts=%d feedback=(likes=%d,skips=%d)",
        decision.is_active,
        decision.adjusted_blend_alpha,
        decision.target_energy,
        len(decision.hard_genre_filters),
        len(decision.hard_mood_filters),
        len(decision.genre_boosts),
        len(decision.mood_boosts),
        likes,
        skips,
    )
    return decision
