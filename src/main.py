import numpy as np
from utils import (
    load_dataset,
    summary,
    bootstrap_sample,
    visualize_results,
    load_dataset_v2,
)
from cosine_lsh import repeated_lsh, run_experiment
from finetuned_sbert import (
    generate_reduced_sbert_embeddings,
    tune_sbert_model,
    generate_tfidf_embeddings,
    generate_count_embeddings,
)
import pandas as pd
from classifier import train_catboost_classifier


np.random.seed(42)


def bootstrap_run(bootstrap_identifier):
    # Temporary change the 500:
    minimal_product_df = load_dataset_v2()

    train_df, test_df, total_train_duplicates, total_test_duplicates = bootstrap_sample(
        minimal_product_df
    )

    print(total_train_duplicates)
    print(total_test_duplicates)

    print("Fine tuning SBERT model...")
    # tune_sbert_model(bootstrap_identifier, train_df)
    print("Fine tuning SBERT finished ✅")

    # Was 96
    train_product_embeddings = generate_reduced_sbert_embeddings(
        16, train_df["title"], bootstrap_identifier
    )
    """
    test_product_embeddings = generate_reduced_sbert_embeddings(
        16, test_df["title"], bootstrap_identifier
    )
    """
    test_product_embeddings = generate_reduced_sbert_embeddings(
        128, test_df["title"], bootstrap_identifier
    )

    # test_product_embeddings = generate_count_embeddings(test_df["title"])

    print("Reduced sentence embeddings generated ✅")

    print("Training catboost model...")
    # Import and train the catboost classifier here
    catboost_model = (
        None  # train_catboost_classifier(train_product_embeddings, train_df)
    )

    print("Catboost model estimated ✅")

    # Random procedure ofcourse values will differ?
    trial_candidates = [1]
    plane_candidates = [8, 12, 16, 24, 36]  # range(3, 48)

    results = run_experiment(
        test_df,
        test_product_embeddings,
        catboost_model,
        trial_candidates,
        plane_candidates,
        total_test_duplicates,
        bootstrap_identifier,
    )

    visualize_results(results)
    return results


all_results = []

for identifier in range(0, 8):
    new_results = bootstrap_run(identifier)
    all_results.extend(new_results)

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
