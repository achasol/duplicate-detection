import numpy as np

# TODO also introduce number of bands


def get_minhash_buckets(hashes):
    buckets = {}
    for index, hash in enumerate(hashes):
        hash_str = "".join(hash.astype(str))
        bucket_id = hash_str
        if bucket_id in buckets:
            buckets[bucket_id].append(index)
        else:
            buckets[bucket_id] = [index]
    return list(buckets.values())


def generate_hash_functions(num_hashes):
    return [
        (np.random.randint(1, 1000), np.random.randint(1, 1000))
        for _ in range(num_hashes)
    ]


def minhash(representation, hash_funcs):
    def hash_value(x, a, b, c=1_000_000_007):
        return (a * x + b) % c

    signature_set = np.full(len(hash_funcs), np.inf)
    for i, (a, b) in enumerate(hash_funcs):
        for element in representation:
            hashed_value = hash_value(element, a, b)
            signature_set[i] = min(signature_set[i], hashed_value)

    return signature_set


def generate_minhashes(num_hashes, embeddings):
    hash_functions = generate_hash_functions(num_hashes)
    hashes = []
    for embedding in embeddings:
        signature = minhash(embedding, hash_functions)
        hashes.append(signature)

    return get_minhash_buckets(hashes)
