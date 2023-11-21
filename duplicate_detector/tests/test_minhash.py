import numpy as np
from ..minhash import (
    hash_row,
    generate_minhash_signature_matrix,
    split_matrix_into_bands,
    lsh_minhash,
    generate_minhashes,
)

# Create sample data for testing
np.random.seed(42)
embeddings = np.rint(np.random.rand(10, 4))


def test_hash_row():
    seed = (123, 456)
    hashed_row = hash_row(embeddings[0], seed)
    assert isinstance(hashed_row, np.ndarray)
    assert hashed_row.shape == (4,)


def test_generate_minhash_signature_matrix():
    num_hashes = 6
    signature_matrix = generate_minhash_signature_matrix(num_hashes, embeddings)
    assert isinstance(signature_matrix, np.ndarray)
    assert signature_matrix.shape == (num_hashes, 10)


def test_split_matrix_into_bands():
    signature_matrix = generate_minhash_signature_matrix(6, embeddings)
    bands = split_matrix_into_bands(signature_matrix, 2, 3)
    assert isinstance(bands, list)
    assert len(bands) == 2
    assert isinstance(bands[0], np.ndarray)
    assert bands[0].shape == (3, 10)


def test_lsh_minhash():
    num_bands = 2
    num_rows = 10
    buckets = lsh_minhash(num_bands, num_rows, embeddings)
    assert isinstance(buckets, list)


def test_generate_minhashes():
    num_bands = 2
    num_rows = 3
    buckets = generate_minhashes(num_bands, num_rows, embeddings)
    assert isinstance(buckets, list)
