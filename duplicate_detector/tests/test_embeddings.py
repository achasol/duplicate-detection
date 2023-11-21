import pytest
import numpy as np

# Assuming the function is in a module called "encoding"

from ..embeddings import one_hot_encode_n_shingles


def test_one_hot_encode_n_shingles():
    # Test case with a list of product titles
    product_titles = ["Product A", "Product B", "Product C"]

    # Set n to 2 for bigrams (shingles of size 2)
    n = 2

    # Call the function
    result_matrix = one_hot_encode_n_shingles(product_titles, n)

    # Check the type of the result
    assert isinstance(result_matrix, np.ndarray)

    print(result_matrix)
    # Check if the one-hot encoding is correct for the given product titles
    expected_encoding = np.array(
        [
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # Product A
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # Product B
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # Product C
        ],
        dtype=int,
    )
    assert np.array_equal(result_matrix, expected_encoding)
