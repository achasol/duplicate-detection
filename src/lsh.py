import numpy as np
from itertools import combinations
from utils import summary
from tqdm import tqdm
from numba import jit

from minhash import generate_minhashes
from scipy.stats import wasserstein_distance


def get_buckets(hashes):
    buckets = {}
    for index, hash in enumerate(hashes):
        hash_str = "".join(hash.astype(str))
        bucket_id = int(hash_str, 2)
        if bucket_id in buckets:
            buckets[bucket_id].append(index)
        else:
            buckets[bucket_id] = [index]
    return list(buckets.values())


def cosine_locality_sensitive_hash(n_planes, vectors):
    vector_dim = len(vectors[0])

    normal_vectors = np.random.randn(vector_dim, n_planes)
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors)
    hashes = (vectors @ normal_vectors > 0).astype(int)
    return get_buckets(hashes)


def jaccard_similarity(embedding1, embedding2):
    cm = np.zeros((2, 2), dtype=int)

    for a, p in zip(embedding1, embedding2):
        cm[a, p] += 1

    return cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0])


def wasserstein_similarity(embedding1, embedding2):
    hamming_distance = np.count_nonzero(embedding1 != embedding2)
    if hamming_distance == 0:
        return 1
    wasserstein_distance_value = wasserstein_distance(embedding1, embedding2)
    return wasserstein_distance_value / hamming_distance


def detect_duplicates(df, buckets, embeddings, already_found, predictions, true_labels):
    true_duplicates = 0
    duplicates_identified = 0
    comparisons_made = 0

    for index, bucket in enumerate(buckets):
        if len(bucket) <= 1:
            continue
        for product1_index, product2_index in combinations(bucket, 2):
            # No counting of duplicates within the same shop
            if df.iloc[product1_index].shop == df.iloc[product2_index].shop:
                continue
            if df.iloc[product1_index].brand != df.iloc[product2_index].brand:
                continue

            # No double counting of duplicates
            if (product1_index, product2_index) in already_found or (
                product2_index,
                product1_index,
            ) in already_found:
                continue

            comparisons_made += 1

            product1 = embeddings[product1_index]
            product2 = embeddings[product2_index]

            prediction = (
                wasserstein_similarity(product1, product2) >= 0.9
                and jaccard_similarity(product1, product2) >= 0.9
            )

            is_duplicate = df.iloc[product1_index].id == df.iloc[product2_index].id

            predictions.append(prediction)
            true_labels.append(is_duplicate)

            true_duplicates += is_duplicate
            decision = is_duplicate and prediction
            duplicates_identified += 1 if decision else 0

            if decision:
                if product1_index < product2_index:
                    already_found[(product1_index, product2_index)] = True
                else:
                    already_found[(product2_index, product1_index)] = True

    return (
        duplicates_identified,
        comparisons_made,
        already_found,
        true_labels,
        predictions,
    )


def repeated_minhash_lsh(num_bands, num_rows, embeddings, df):
    duplicate_pairs_found = {}
    duplicates_identified = 0
    comparisons_made = 0
    predictions = []
    true_labels = []

    buckets = generate_minhashes(num_bands, num_rows, embeddings)

    (
        new_duplicates_identified,
        new_comparisons_made,
        duplicate_pairs_found,
        predictions,
        true_labels,
    ) = detect_duplicates(
        df,
        buckets,
        embeddings,
        duplicate_pairs_found,
        predictions,
        true_labels,
    )

    duplicates_identified += new_duplicates_identified
    comparisons_made += new_comparisons_made

    return duplicates_identified, comparisons_made, predictions, true_labels


def run_experiment(
    df,
    embeddings,
    row_candidates,
    band_candidates,
    total_duplicates,
    run_identifier,
):
    results = []

    for num_bands in tqdm(band_candidates):
        for num_rows in tqdm(row_candidates):
            (
                duplicates_identified,
                comparisons_made,
                predictions,
                true_labels,
            ) = repeated_minhash_lsh(num_bands, num_rows, embeddings, df)

            (
                pair_quality,
                pair_completeness,
                f1_star_score,
                f1score,
                fraction_comparisons,
            ) = summary(
                duplicates_identified,
                total_duplicates,
                comparisons_made,
                len(df),
                predictions,
                true_labels,
                print_output=False,
            )

            results.append(
                [
                    round(elem, 3)
                    for elem in [
                        run_identifier,
                        num_bands,
                        num_rows,
                        pair_quality,
                        pair_completeness,
                        f1_star_score,
                        f1score,
                        fraction_comparisons,
                    ]
                ]
            )

    return results