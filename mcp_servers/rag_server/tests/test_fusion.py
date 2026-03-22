import pytest

from mcp_servers.rag_server.fusion import order_fusion_scores, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_basic_weights():
    rank_lists = [
        ["chunk_a", "chunk_b", "chunk_c"],
        ["chunk_b", "chunk_a", "chunk_d"],
    ]

    fused = reciprocal_rank_fusion(rank_lists, k=60)

    assert fused["chunk_a"] == pytest.approx((1 / 61) + (1 / 62))
    assert fused["chunk_b"] == pytest.approx((1 / 62) + (1 / 61))
    assert fused["chunk_c"] == pytest.approx(1 / 63)


def test_order_fusion_scores_tie_breaker_is_deterministic():
    scores = {
        "z_chunk": 0.5,
        "a_chunk": 0.5,
        "m_chunk": 0.25,
    }
    ranked = order_fusion_scores(scores, top_k=2)

    assert ranked[0][0] == "a_chunk"
    assert ranked[1][0] == "z_chunk"
