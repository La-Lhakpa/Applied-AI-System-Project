"""
CLI demo for the EchoMind recommendation pipeline.

Runs four contrasting taste profiles through the blended pipeline and
prints ranked results with decomposed scores, showing how different users
receive different recommendations from the same catalog.

Usage:
    python -m src.main
"""

from src.pipeline import RecommendationPipeline, RecommendationResult
from src.recommender import Song, UserProfile, load_songs

_WIDE = "=" * 65
_THIN = "-" * 65

DEMO_PROFILES = [
    (
        "Happy Pop Fan",
        UserProfile(favorite_genre="pop", favorite_mood="happy",
                    target_energy=0.8, likes_acoustic=False),
    ),
    (
        "Chill Lofi Listener",
        UserProfile(favorite_genre="lofi", favorite_mood="chill",
                    target_energy=0.4, likes_acoustic=True),
    ),
    (
        "High-Energy Rocker",
        UserProfile(favorite_genre="rock", favorite_mood="intense",
                    target_energy=0.9, likes_acoustic=False),
    ),
    (
        "Laid-Back Jazz Lover",
        UserProfile(favorite_genre="jazz", favorite_mood="relaxed",
                    target_energy=0.35, likes_acoustic=True),
    ),
]


def _print_result(rank: int, r: RecommendationResult) -> None:
    song = r.song
    print(
        f"  {rank}. {song.title:<28} {song.artist:<18} "
        f"[{song.genre:<10} / {song.mood:<12}]"
    )
    print(
        f"     Score: {r.final_score:.3f}  "
        f"(content={r.content_score:.3f}  label={r.label_score:.3f}  "
        f"energy={song.energy:.2f})"
    )


def main() -> None:
    raw = load_songs("data/songs.csv")
    songs = [Song(**s) for s in raw]

    print(f"\n{_WIDE}")
    print("  EchoMind — Blended Recommendation Pipeline Demo")
    print("  Signal mix: 50% content-based  +  50% label-based")
    print(_WIDE)

    pipeline = RecommendationPipeline(songs, blend_alpha=0.5)

    for name, profile in DEMO_PROFILES:
        print(f"\n{'Profile'}: {name}")
        print(
            f"  genre={profile.favorite_genre!r}  "
            f"mood={profile.favorite_mood!r}  "
            f"energy={profile.target_energy}  "
            f"acoustic={profile.likes_acoustic}"
        )
        print(_THIN)
        results = pipeline.run(profile, k=5)
        for i, r in enumerate(results, 1):
            _print_result(i, r)
        print()

    print(_WIDE)
    print("  Tip: run  streamlit run src/app.py  for the interactive UI")
    print(_WIDE)


if __name__ == "__main__":
    main()
