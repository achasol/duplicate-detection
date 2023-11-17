import numpy as np
from itertools import combinations
from classifier import generate_catboost_sample
from utils import summary
from tqdm import tqdm
from numba import jit
import math
from minhash import generate_minhashes

# Generate an array of buckets using the generated hashcodes


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


# Implementation of cosine LSH using random hyperplanes
def cosine_locality_sensitive_hash(n_planes, vectors):
    vector_dim = len(vectors[0])

    normal_vectors = np.random.randn(vector_dim, n_planes)
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors)
    hashes = (vectors @ normal_vectors > 0).astype(int)
    return get_buckets(hashes)


# Method which iterates over the LSH buckets and uses the classifier to determine which elements are duplicates.
# TODO bug in the method a product id can be found more than 2 times (multi duplicate!!)
# Instead add sorted index pair i < j as identifiers to already_found!
def detect_duplicates(
    df, buckets, embeddings, already_found, duplicate_detector, predictions, true_labels
):
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

            catboost_sample = generate_catboost_sample(product1, product2)
            prediction = (
                df.iloc[product1_index].id == df.iloc[product2_index].id
            )  # duplicate_detector.predict(catboost_sample)

            is_duplicate = df.iloc[product1_index].id == df.iloc[product2_index].id

            predictions.append(prediction)
            true_labels.append(is_duplicate)

            true_duplicates += is_duplicate
            decision = (
                is_duplicate and np.rint(prediction) == 1
            )  # Needs to change currently perfect measure
            duplicates_identified += 1 if decision else 0

            if decision:
                if product1_index < product2_index:
                    already_found[(product1_index, product2_index)] = True
                else:
                    already_found[(product2_index, product1_index)] = True

    # print(
    #    "comparisons performed: "
    #    + str(comparisons_made)
    #    + "/"
    #    + str(sum([math.comb(len(bucket), 2) for bucket in buckets]))
    # )

    return (
        duplicates_identified,
        comparisons_made,
        already_found,
        true_labels,
        predictions,
    )


def repeated_minhash_lsh(
    num_hashes, num_bands, num_rows, embeddings, df, duplicate_detector
):
    duplicate_pairs_found = {}
    duplicates_identified = 0
    comparisons_made = 0
    predictions = []
    true_labels = []

    # Preserve this equality at all times
    num_hashes = num_bands * num_rows
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
        duplicate_detector,
        predictions,
        true_labels,
    )

    duplicates_identified += new_duplicates_identified
    comparisons_made += new_comparisons_made

    return duplicates_identified, comparisons_made, predictions, true_labels


# Method which repeats the cosine lsh multiple times to find more duplicates.
def repeated_lsh(trials, random_planes, embeddings, df, duplicate_detector):
    duplicate_pairs_found = {}
    duplicates_identified = 0
    comparisons_made = 0
    predictions = []
    true_labels = []

    for t in range(0, trials):
        buckets = generate_minhashes(random_planes, embeddings)

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
            duplicate_detector,
            predictions,
            true_labels,
        )

        duplicates_identified += new_duplicates_identified
        comparisons_made += new_comparisons_made

    return duplicates_identified, comparisons_made, predictions, true_labels


def run_experiment(
    df,
    embeddings,
    duplicate_detector,
    row_candidates,
    band_candidates,
    num_hashes,
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
            ) = repeated_minhash_lsh(
                num_hashes, num_bands, num_rows, embeddings, df, duplicate_detector
            )

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
