"""
Implement hashing based on cosine LSH 
"""
import numpy as np


def get_buckets(hashes):
    buckets = {}
    for index, hash_val in enumerate(hashes):
        if str(hash_val) in buckets:
            buckets[str(hash_val)].append(index)
        else:
            buckets[str(hash_val)] = [index]

    return buckets


def generate_cosine_hashes(num_trials, num_planes, embeddings):
    all_buckets = []
    for trial in range(num_trials):
        random_planes = np.random.standard_normal((num_planes, len(embeddings[0])))

        hashes = (embeddings @ random_planes.T > 0).astype(int)
        all_buckets.append(get_buckets(hashes))

    unique_buckets = get_unique_buckets(all_buckets)

    bucket_sizes = [len(bucket) for bucket in unique_buckets]

    # Drop the 10% largest buckets
    threshold = np.percentile(bucket_sizes, 95)  # 85

    top_indices = np.where(bucket_sizes >= threshold)[0]
    unique_buckets = list(unique_buckets)
    # TODO figure out weakness of minhash implementation
    # print(sorted([len(bucket) for bucket in unique_buckets]))
    unique_trimmed_buckets = [
        value for index, value in enumerate(unique_buckets) if index not in top_indices
    ]

    return unique_trimmed_buckets


def get_unique_buckets(buckets):
    unique_buckets = set()

    for bucket in buckets:
        for value in bucket.values():
            if len(value) == 1:
                continue
            unique_buckets.add(tuple(value))

    return unique_buckets
