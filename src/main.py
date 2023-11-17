import numpy as np
from utils import (
    bootstrap_sample,
    visualize_results,
    load_dataset_v2,
)
from cosine_lsh import run_experiment
from finetuned_sbert import (
    generate_reduced_sbert_embeddings,
    one_hot_encode_n_shingles,
)
import pandas as pd

# from classifier import train_catboost_classifier
from concurrent.futures import ProcessPoolExecutor


def bootstrap_run(bootstrap_identifier):
    # Temporary change the 500:
    np.random.seed(42 + bootstrap_identifier)
    minimal_product_df = load_dataset_v2()

    train_df, test_df, total_train_duplicates, total_test_duplicates = bootstrap_sample(
        minimal_product_df
    )

    # Was 96
    train_product_embeddings = generate_reduced_sbert_embeddings(
        16, train_df["title"], bootstrap_identifier
    )
    """
    test_product_embeddings = generate_reduced_sbert_embeddings(
        16, test_df["title"], bootstrap_identifier
    )
    """

    test_product_embeddings = one_hot_encode_n_shingles(test_df["title"], 8)  # 8

    catboost_model = (
        None  # train_catboost_classifier(train_product_embeddings, train_df)
    )

    # print("Catboost model estimated ✅")

    num_hashes = 4  # Parameter has no real effect (computed based  on rows and bands)
    row_candidates = [20]  # [25, 50]  # range(25,50)
    band_candidates = [5]  # range(1, 3)

    results = run_experiment(
        test_df,
        test_product_embeddings,
        catboost_model,
        row_candidates,
        band_candidates,
        num_hashes,
        total_test_duplicates,
        bootstrap_identifier,
    )

    visualize_results(results)
    return results


def bootstrap_run_parallel(identifier):
    return bootstrap_run(identifier)


def process_results(all_results):
    df = pd.DataFrame(
        all_results,
        columns=[
            "run_identifier",
            "n_trial",
            "n_planes",
            "pair_quality",
            "pair_completeness",
            "f1*_score",
            "f1_score",
            "comparisons_fraction",
        ],
    )
    df.to_csv(r"../results/bootstrap-runs.csv", index=None)


if __name__ == "__main__":
    all_results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Use executor.map to run bootstrap_run in parallel
        results_list = list(executor.map(bootstrap_run_parallel, range(8)))
        for new_results in results_list:
            all_results.extend(new_results)

    process_results(all_results)
