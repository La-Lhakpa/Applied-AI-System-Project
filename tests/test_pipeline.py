from src.recommender import Song, UserProfile
from src.pipeline import RecommendationPipeline, RecommendationResult


def _catalog() -> list:
    return [
        Song(id=1, title="Pop Hit",    artist="A", genre="pop",  mood="happy",
             energy=0.82, tempo_bpm=118, valence=0.84, danceability=0.79, acousticness=0.18),
        Song(id=2, title="Chill Lofi", artist="B", genre="lofi", mood="chill",
             energy=0.42, tempo_bpm=78,  valence=0.56, danceability=0.62, acousticness=0.71),
        Song(id=3, title="Rock Storm", artist="C", genre="rock", mood="intense",
             energy=0.91, tempo_bpm=152, valence=0.48, danceability=0.66, acousticness=0.10),
    ]


def _pop_user() -> UserProfile:
    return UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)


def test_pipeline_returns_k_results():
    results = RecommendationPipeline(_catalog()).run(_pop_user(), k=2)
    assert len(results) == 2


def test_pipeline_returns_recommendation_result_instances():
    results = RecommendationPipeline(_catalog()).run(_pop_user(), k=1)
    assert isinstance(results[0], RecommendationResult)


def test_results_sorted_descending_by_final_score():
    results = RecommendationPipeline(_catalog()).run(_pop_user(), k=3)
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_explanation_is_non_empty_string():
    results = RecommendationPipeline(_catalog()).run(_pop_user(), k=1)
    assert isinstance(results[0].explanation, str)
    assert results[0].explanation.strip() != ""


def test_alpha_zero_equals_label_only():
    pipeline = RecommendationPipeline(_catalog(), blend_alpha=0.0)
    for r in pipeline.run(_pop_user(), k=3):
        assert abs(r.final_score - r.label_score) < 1e-9


def test_alpha_one_equals_content_only():
    pipeline = RecommendationPipeline(_catalog(), blend_alpha=1.0)
    for r in pipeline.run(_pop_user(), k=3):
        assert abs(r.final_score - r.content_score) < 1e-9


def test_scores_in_unit_range():
    pipeline = RecommendationPipeline(_catalog())
    for r in pipeline.run(_pop_user(), k=3):
        assert 0.0 <= r.content_score <= 1.0
        assert 0.0 <= r.label_score <= 1.0


def test_pop_user_top_result_is_pop_song():
    results = RecommendationPipeline(_catalog()).run(_pop_user(), k=1)
    assert results[0].song.genre == "pop"
