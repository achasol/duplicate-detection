import numpy as np
from numba import njit

# https://planetmath.org/goodhashtableprimes


@njit
def hash_row(row, seed):
    """
    Hash a given row using the specified seed.

    Parameters:
    - row (numpy.ndarray): The input row to be hashed.
    - seed (tuple): A tuple of two integers used as the seed for hashing.

    Returns:
    numpy.ndarray: The hashed row.
    """
    a, b = seed
    c = 402653189
    return np.mod(a + b * row, c)


def generate_minhash_signature_matrix(num_hashes, embeddings):
    """
    Generate a MinHash signature matrix for the given embeddings.

    Parameters:
    - num_hashes (int): The number of hash functions to be used.
    - embeddings (numpy.ndarray): The input embeddings.

    Returns:
    numpy.ndarray: The MinHash signature matrix.
    """
    embeddings = embeddings.T
    signature_matrix = np.array(np.ones((num_hashes, len(embeddings[0]))) * np.inf)

    for row in embeddings:
        hashed_row = []
        seeds = np.random.randint(1, 2**24, size=(num_hashes, 2))

        for seed in seeds:
            hashed_row.append(hash_row(row, seed))

        for index, c in enumerate(row):
            if c == 0:
                continue
            for i in range(num_hashes):
                signature_matrix[i, index] = min(
                    signature_matrix[i, index],
                    min(hashed_row[i]),
                )

    signatures = signature_matrix
    return signatures


def split_matrix_into_bands(signature_matrix, b, r):
    """
    Split the signature matrix into bands.

    Parameters:
    - signature_matrix (numpy.ndarray): The input MinHash signature matrix.
    - b (int): The number of bands.
    - r (int): The number of rows per band.

    Returns:
    list: A list of bands, where each band is a submatrix of the signature matrix.
    """
    num_rows, num_cols = signature_matrix.shape

    if num_rows % b != 0:
        raise ValueError(f"Number of rows ({num_rows}) is not divisible by b ({b}).")

    band_size = num_rows // b
    bands = []

    for i in range(b):
        start_row = i * band_size
        end_row = (i + 1) * band_size
        band = signature_matrix[start_row:end_row, :]
        bands.append(band)

    return bands


def lsh_minhash(num_bands, num_rows, embeddings):
    """
    Perform Locality-Sensitive Hashing (LSH) on MinHash signatures.

    Parameters:
    - num_bands (int): The number of bands.
    - num_rows (int): The number of rows per band.
    - embeddings (numpy.ndarray): The input embeddings.

    Returns:
    list: A list of unique buckets representing candidate similar items.
    """
    num_hashes = num_bands * num_rows
    signatures = generate_minhash_signature_matrix(num_hashes, embeddings)
    buckets = [{} for band in range(num_bands * num_rows)]

    candidates = split_matrix_into_bands(signatures, num_bands, num_rows)

    for index, candidate in enumerate(candidates):
        for embedding_index, row in enumerate(candidate.T):
            hash_str = hash(str(row))
            if str(hash_str) in buckets[index]:
                buckets[index][str(hash_str)].append(embedding_index)
            else:
                buckets[index][str(hash_str)] = [embedding_index]

    unique_buckets = set()
    for bucket in buckets:
        for value in bucket.values():
            if len(value) == 1:
                continue
            unique_buckets.add(tuple(value))

    bucket_sizes = [len(bucket) for bucket in unique_buckets]

    # Drop the 10% largest buckets
    threshold = np.percentile(bucket_sizes, 85)  # 85

    top_indices = np.where(bucket_sizes >= threshold)[0]
    unique_buckets = list(unique_buckets)
    unique_trimmed_buckets = [
        value for index, value in enumerate(unique_buckets) if index not in top_indices
    ]
    return unique_trimmed_buckets


def generate_minhashes(num_bands, num_rows, embeddings):
    """
    Generate MinHashes using Locality-Sensitive Hashing.

    Parameters:
    - num_bands (int): The number of bands.
    - num_rows (int): The number of rows per band.
    - embeddings (numpy.ndarray): The input embeddings.

    Returns:
    list: A list of unique buckets representing candidate similar items.
    """
    return lsh_minhash(num_bands, num_rows, embeddings)
