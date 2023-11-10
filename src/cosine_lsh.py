import numpy as np
from itertools import combinations
from classifier import generate_catboost_sample
from utils import summary
from tqdm import tqdm


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
def detect_duplicates(df, buckets, embeddings, already_found, duplicate_detector):
    true_duplicates = 0
    duplicates_identified = 0
    comparisons_made = 0

    for bucket in buckets:
        if len(bucket) <= 1:
            continue
        for product1_index, product2_index in combinations(bucket, 2):
            if df.iloc[product1_index].shop == df.iloc[product2_index].shop:
                continue
            if df.iloc[product1_index].id in already_found:
                continue

            comparisons_made += 1

            product1 = embeddings[product1_index]
            product2 = embeddings[product2_index]

            catboost_sample = generate_catboost_sample(product1, product2)
            prediction = duplicate_detector.predict(catboost_sample)

            is_duplicate = df.iloc[product1_index].id == df.iloc[product2_index].id

            true_duplicates += is_duplicate
            decision = (
                is_duplicate and np.rint(prediction) == 1
            )  # Needs to change currently perfect measure
            duplicates_identified += 1 if decision else 0

            if decision:
                already_found[df.iloc[product1_index].id] = True

    return duplicates_identified, comparisons_made, already_found


# Method which repeats the cosine lsh multiple times to find more duplicates.
def repeated_lsh(trials, random_planes, embeddings, df, duplicate_detector):
    duplicate_pairs_found = {}
    duplicates_identified = 0
    comparisons_made = 0

    for t in range(0, trials):
        buckets = cosine_locality_sensitive_hash(random_planes, embeddings)

        (
            new_duplicates_identified,
            new_comparisons_made,
            duplicate_pairs_found,
        ) = detect_duplicates(
            df, buckets, embeddings, duplicate_pairs_found, duplicate_detector
        )

        duplicates_identified += new_duplicates_identified
        comparisons_made += new_comparisons_made

    return duplicates_identified, comparisons_made


def run_experiment(
    df,
    embeddings,
    duplicate_detector,
    trial_candidates,
    plane_candidates,
    total_duplicates,
    run_identifier,
):
    results = []
    best_f1score = 0
    for N_TRIALS in tqdm(trial_candidates):
        for N_PLANES in tqdm(plane_candidates):
            duplicates_identified, comparisons_made = repeated_lsh(
                N_TRIALS, N_PLANES, embeddings, df, duplicate_detector
            )

            pair_quality, pair_completeness, f1score, fraction_comparisons = summary(
                duplicates_identified,
                total_duplicates,
                comparisons_made,
                len(df),
                print_output=False,
            )

            results.append(
                [
                    round(elem, 3)
                    for elem in [
                        run_identifier,
                        N_TRIALS,
                        N_PLANES,
                        pair_quality,
                        pair_completeness,
                        f1score,
                        fraction_comparisons,
                    ]
                ]
            )

            if f1score > best_f1score:
                best_f1score = f1score
                print(
                    f"Best: identifier {run_identifier} - planes - {N_PLANES} - trials - {N_TRIALS} - f1 - {f1score} - {fraction_comparisons}"
                )
    return results
