from .utils import bootstrap_sample, visualize_results, load_dataset, process_results
from .lsh import run_experiment
import numpy as np
from .embeddings import one_hot_encode_n_shingles
from concurrent.futures import ProcessPoolExecutor


def run():
    all_results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Use executor.map to run bootstrap_run in parallel
        results_list = list(executor.map(bootstrap_run_parallel, range(8)))
        for new_results in results_list:
            all_results.extend(new_results)

    process_results(all_results)


def bootstrap_run(bootstrap_identifier):
    """
    Perform a single run of the bootstrap sampling and LSH experiment.

    Parameters:
    - bootstrap_identifier (int): Identifier for the bootstrap run.

    Returns:
    list: Results of the LSH experiment for the given bootstrap run.
    """
    np.random.seed(42 + bootstrap_identifier)
    minimal_product_df = load_dataset()

    train_df, test_df, total_train_duplicates, total_test_duplicates = bootstrap_sample(
        minimal_product_df
    )

    # train_product_embeddings
    test_product_embeddings = one_hot_encode_n_shingles(test_df["title"], 8)  # 8

    row_candidates = [20]  # range(1, 64)  # [25, 50]  # range(25,50)
    band_candidates = [5]  # range(1, 64)  # range(1, 3)

    results = run_experiment(
        test_df,
        test_product_embeddings,
        row_candidates,
        band_candidates,
        total_test_duplicates,
        bootstrap_identifier,
    )

    visualize_results(results)
    return results


def bootstrap_run_parallel(identifier):
    """
    Perform a single run of the bootstrap sampling and LSH experiment in parallel.

    Parameters:
    - identifier (int): Identifier for the parallel run.

    Returns:
    list: Results of the LSH experiment for the given parallel run.
    """
    return bootstrap_run(identifier)
