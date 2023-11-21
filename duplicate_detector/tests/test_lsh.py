import numpy as np
import pandas as pd
import pytest
from ..lsh import (
    get_buckets,
    jaccard_similarity,
    wasserstein_similarity,
    detect_duplicates,
    repeated_minhash_lsh,
    run_experiment,
)


@pytest.fixture
def example_data():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "shop": ["A", "B", "A", "B", "C"],
            "brand": ["X", "Y", "X", "Y", "Z"],
        }
    )

    embeddings = np.random.rand(5, 10)  # Example embeddings
    row_candidates = [2, 4]
    band_candidates = [2, 4]

    return df, embeddings, row_candidates, band_candidates, 2, "test_run"


def test_get_buckets():
    hashes = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    buckets = get_buckets(hashes)
    assert len(buckets) == 2
    assert set(buckets[0]) == {0, 2}
    assert set(buckets[1]) == {1}


def test_jaccard_similarity():
    embedding1 = np.array([0, 1, 1, 0])
    embedding2 = np.array([1, 1, 0, 0])
    similarity = jaccard_similarity(embedding1, embedding2)
    assert similarity == 1 / 3


def test_wasserstein_similarity():
    embedding1 = np.array([0, 1, 1, 0])
    embedding2 = np.array([1, 1, 0, 0])
    similarity = wasserstein_similarity(embedding1, embedding2)
    assert 0 <= similarity <= 1


def test_detect_duplicates(example_data):
    df, embeddings, _, _, _, _ = example_data
    buckets = [[0, 1], [2, 3]]
    already_found = {}
    predictions = []
    true_labels = []

    result = detect_duplicates(
        df, buckets, embeddings, already_found, predictions, true_labels
    )

    assert len(result) == 5  # Check the expected number of return values
    # Add more assertions based on the expected behavior of detect_duplicates


def test_repeated_minhash_lsh(example_data):
    _, embeddings, _, _ = example_data[:4]
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "shop": ["A", "B", "A", "B", "C"],
            "brand": ["X", "Y", "X", "Y", "Z"],
        }
    )

    result = repeated_minhash_lsh(2, 2, embeddings, df)

    assert len(result) == 4  # Check the expected number of return values
    # Add more assertions based on the expected behavior of repeated_minhash_lsh
