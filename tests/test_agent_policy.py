from src.agent_policy import SessionFeedback, decide_policy
from src.recommender import UserProfile


def _user() -> UserProfile:
    return UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.7,
        likes_acoustic=False,
    )


def test_policy_no_signal_is_inactive_noop():
    policy = decide_policy(_user(), default_blend_alpha=0.5, intent_text="", session_feedback=None)
    assert policy.is_active is False
    assert abs(policy.adjusted_blend_alpha - 0.5) < 1e-12
    assert abs(policy.target_energy - 0.7) < 1e-12


def test_policy_detects_high_energy_intent():
    policy = decide_policy(_user(), default_blend_alpha=0.5, intent_text="high-energy workout mix")
    assert policy.is_active is True
    assert policy.adjusted_blend_alpha > 0.5
    assert policy.target_energy > 0.7
    assert policy.energy_proximity_weight > 0


def test_feedback_polarity_adjusts_blend_alpha():
    positive = decide_policy(
        _user(), default_blend_alpha=0.5, session_feedback=SessionFeedback(likes=8, skips=2)
    )
    negative = decide_policy(
        _user(), default_blend_alpha=0.5, session_feedback=SessionFeedback(likes=2, skips=8)
    )
    assert positive.adjusted_blend_alpha > negative.adjusted_blend_alpha
