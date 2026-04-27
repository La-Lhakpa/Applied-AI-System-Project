"""
Similarity engine: pure-Python vector math for content-based scoring.

Cosine similarity measures the angle between two feature vectors — a value
of 1.0 means identical taste direction, 0.0 means orthogonal (unrelated).
This is the core of audio-feature-based recommendation (Spotify's "audio
fingerprint" approach) as opposed to purely label-matching systems.
"""

import math
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two equal-length float vectors.

    Returns a value in [0, 1] for non-negative feature spaces.
    Returns 0.0 if either vector has zero magnitude.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def rank_by_similarity(
    taste_vector: List[float],
    song_vectors: List[List[float]],
) -> List[float]:
    """
    Compute cosine similarity between the taste vector and every song vector.

    Returns a list of float scores aligned with the input song_vectors list.
    """
    return [cosine_similarity(taste_vector, sv) for sv in song_vectors]
