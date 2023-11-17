"""
Write a complete minhash implementation from scratch 
"""
import numpy as np
from numba import njit

shingles = np.array(
    [
        [1, 1, 0, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
)


# https://planetmath.org/goodhashtableprimes

"""
@njit
def hash_function(a: int, b: int, x: int, c=3145739):  # 786433
    return np.mod(a + b * x, c)


def hash_row(row, seed):
    return [hash_function(seed[0], seed[1], c) for c in row]
"""


@njit
def hash_row(row, seed):
    a, b = seed
    c = 402653189
    return np.mod(a + b * row, c)


def generate_minhash_signature_matrix(num_hashes, embeddings):
    embeddings = embeddings.T
    signature_matrix = np.matrix(np.ones((num_hashes, len(embeddings[0]))) * np.inf)
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
                    min(hashed_row[i]),  # Or min(hashed_row[i])
                )

    signatures = signature_matrix

    return signatures


def split_matrix_into_bands(signature_matrix, b, r):
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


"""
Need to know which bands and rows match to which shingles 
"""


def lsh_minhash(num_bands, num_rows, embeddings):
    num_hashes = num_bands * num_rows
    signatures = generate_minhash_signature_matrix(num_hashes, embeddings)
    buckets = [{} for band in range(num_bands * num_rows)]

    candidates = split_matrix_into_bands(signatures, num_bands, num_rows)

    for index, candidate in enumerate(candidates):
        for embedding_index, row in enumerate(candidate.T):
            hash_str = hash(str(row))  # hash_row(row, (1200, 1000))[0].tolist()
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
    threshold = np.percentile(bucket_sizes, 85)

    top_indices = np.where(bucket_sizes >= threshold)[0]
    unique_buckets = list(unique_buckets)
    unique_trimmed_buckets = [
        value for index, value in enumerate(unique_buckets) if index not in top_indices
    ]
    return unique_trimmed_buckets


def generate_minhashes(num_bands, num_rows, embeddings):
    return lsh_minhash(num_bands, num_rows, embeddings)


# lsh_minhash(4, 3, shingles)
