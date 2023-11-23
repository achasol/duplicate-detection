import numpy as np
from itertools import combinations
from .utils import summary
from tqdm import tqdm
from numba import jit

from .minhash import generate_minhashes

from .cosinehash import generate_cosine_hashes


def get_buckets(hashes):
    """
    Assign vectors to buckets based on their hash values.

    Parameters:
    - hashes (numpy.ndarray): Array of hash values for each vector.

    Returns:
    list: List of buckets, where each bucket contains indices of vectors sharing the same hash value.
    """
    buckets = {}
    for index, hash in enumerate(hashes):
        hash_str = "".join(hash.astype(str))
        bucket_id = int(hash_str, 2)
        if bucket_id in buckets:
            buckets[bucket_id].append(index)
        else:
            buckets[bucket_id] = [index]
    return list(buckets.values())


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding1)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm1 == 0:
        return 0

    similarity = dot_product / (norm1 * norm2)
    return similarity


def jaccard_similarity(embedding1, embedding2):
    """
    Calculate Jaccard similarity between two embeddings.

    Parameters:
    - embedding1 (numpy.ndarray): First embedding vector.
    - embedding2 (numpy.ndarray): Second embedding vector.

    Returns:
    float: Jaccard similarity between the two embeddings.
    """
    cm = np.zeros((2, 2), dtype=int)

    for a, p in zip(embedding1, embedding2):
        cm[a, p] += 1

    if (cm[1, 1] + cm[0, 1] + cm[1, 0]) == 0:
        return 0

    return cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0])


def hamming_similarity(embedding1, embedding2):
    """
    Calculate Wasserstein similarity between two embeddings.

    Parameters:
    - embedding1 (numpy.ndarray): First embedding vector.
    - embedding2 (numpy.ndarray): Second embedding vector.

    Returns:
    float: Wasserstein similarity between the two embeddings.
    """
    hamming_distance = np.count_nonzero(embedding1 != embedding2)
    if hamming_distance == 0:
        return 1

    return hamming_distance / len(embedding1)


def detect_duplicates(df, buckets, embeddings, already_found, predictions, true_labels):
    """
    Detect duplicate pairs within buckets and update statistics.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing product information.
    - buckets (list): List of buckets containing indices of similar vectors.
    - embeddings (numpy.ndarray): Embeddings of the vectors.
    - already_found (dict): Dictionary to track already identified duplicates.
    - predictions (list): List to store duplicate predictions.
    - true_labels (list): List to store true duplicate labels.

    Returns:
    tuple: Tuple containing statistics on duplicates identified, comparisons made, and updated tracking information.
    """
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
                hamming_similarity(product1, product2) >= 0.9
                and jaccard_similarity(product1, product2) >= 0.9
                # cosine_similarity(product1, product2)
                # >= 0.9
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
    """
    Apply repeated MinHash Locality-Sensitive Hashing to identify duplicate pairs.

    Parameters:
    - num_bands (int): Number of bands for LSH.
    - num_rows (int): Number of rows per band.
    - embeddings (numpy.ndarray): Embeddings of the vectors.
    - df (pandas.DataFrame): DataFrame containing product information.

    Returns:
    tuple: Tuple containing statistics on duplicates identified, comparisons made, and prediction information.
    """
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
    """
    Run an experiment with varying numbers of bands and rows, and record the results.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing product information.
    - embeddings (numpy.ndarray): Embeddings of the vectors.
    - row_candidates (list): List of candidate values for the number of rows.
    - band_candidates (list): List of candidate values for the number of bands.
    - total_duplicates (int): Total number of true duplicates in the dataset.
    - run_identifier (str): Identifier for the experiment run.

    Returns:
    list: List of results for each combination of bands and rows.
    """
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
                    round(elem, 12)
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
