import torch
import numpy as np

from rope import apply_rotary_emb

seed = 0


def construct_query() -> torch.Tensor:
    """
    Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
    """
    return 2 * torch.ones([1, 2, 2, 4])


def construct_key() -> torch.Tensor:
    """
    Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
    """
    return 3 * torch.ones([1, 2, 2, 4])


def test_apply_rotary_emb() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    test_query = construct_query()
    test_key = construct_key()
    rotary_embeddings = apply_rotary_emb(test_query, test_key, 4, 20)
    rotary_query_embedding, rotary_key_embedding = rotary_embeddings
    return rotary_query_embedding, rotary_key_embedding


actual_query_rope_embedding, actual_key_rope_embedding = test_apply_rotary_emb()
ref_query_rope_embedding, ref_key_rope_embedding = torch.load(
    "./rotary_embedding_actual.data"
)
assert (
    actual_query_rope_embedding.shape == ref_query_rope_embedding.shape
), f"Query shape didn't match. Expected: {ref_query_rope_embedding.shape}, but got {actual_query_rope_embedding.shape}"
assert (
    actual_key_rope_embedding.shape == ref_key_rope_embedding.shape
), f"Key shape didn't match. Expected: {ref_key_rope_embedding.shape}, but got {actual_key_rope_embedding.shape}"

assert torch.allclose(ref_query_rope_embedding, actual_query_rope_embedding), (
    "Expected query: ",
    ref_query_rope_embedding,
    "Actual query: ",
    actual_query_rope_embedding,
)
assert torch.allclose(ref_key_rope_embedding, actual_key_rope_embedding), (
    "Expected key: ",
    ref_key_rope_embedding,
    "Actual key: ",
    actual_key_rope_embedding,
)
print("Rotary embedding test passed!")
