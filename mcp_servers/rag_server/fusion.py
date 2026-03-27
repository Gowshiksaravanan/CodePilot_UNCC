from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """
    Combine ranked result lists with Reciprocal Rank Fusion (RRF).
    Higher score means higher combined relevance.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return scores


def order_fusion_scores(scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    """
    Return chunk ids sorted by descending score, deterministic on ties.
    """
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]

