from src.recommender import Song, UserProfile
from src.features import FeatureExtractor


def _song(**kwargs) -> Song:
    defaults = dict(
        id=1, title="T", artist="A", genre="pop", mood="happy",
        energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2,
    )
    defaults.update(kwargs)
    return Song(**defaults)


def test_song_vector_correct_length():
    v = FeatureExtractor().song_vector(_song())
    assert len(v) == len(FeatureExtractor.FEATURE_NAMES)


def test_song_vector_all_in_unit_range():
    v = FeatureExtractor().song_vector(
        _song(energy=0.8, valence=0.9, danceability=0.8, acousticness=0.2, tempo_bpm=120)
    )
    for val in v:
        assert 0.0 <= val <= 1.0, f"{val} out of [0, 1]"


def test_tempo_extremes_clamped():
    low = FeatureExtractor().song_vector(_song(tempo_bpm=60))
    high = FeatureExtractor().song_vector(_song(tempo_bpm=180))
    assert low[-1] == 0.0
    assert high[-1] == 1.0


def test_profile_vector_correct_length():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)
    v = FeatureExtractor().profile_vector(user)
    assert len(v) == len(FeatureExtractor.FEATURE_NAMES)


def test_profile_vector_acoustic_flag():
    ex = FeatureExtractor()
    v_yes = ex.profile_vector(UserProfile("pop", "happy", 0.7, True))
    v_no = ex.profile_vector(UserProfile("pop", "happy", 0.7, False))
    # acousticness is index 3
    assert v_yes[3] > v_no[3]


def test_profile_vector_energy_reflects_target():
    ex = FeatureExtractor()
    v = ex.profile_vector(UserProfile("pop", "happy", 0.9, False))
    assert v[0] == 0.9


def test_all_song_vectors_length_matches_catalog():
    songs = [_song(id=i) for i in range(5)]
    vectors = FeatureExtractor().all_song_vectors(songs)
    assert len(vectors) == 5
