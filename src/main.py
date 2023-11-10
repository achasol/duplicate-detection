import numpy as np
from utils import load_dataset, summary, bootstrap_sample, visualize_results
from cosine_lsh import repeated_lsh, run_experiment
from finetuned_sbert import generate_reduced_sbert_embeddings, tune_sbert_model
import pandas as pd
from classifier import train_catboost_classifier

np.random.seed(42)


def bootstrap_run(bootstrap_identifier):
    # Temporary change the 500:
    minimal_product_df = load_dataset()

    train_df, test_df, total_train_duplicates, total_test_duplicates = bootstrap_sample(
        minimal_product_df
    )

    print("Fine tuning SBERT model...")
    tune_sbert_model(bootstrap_identifier, train_df)
    print("Fine tuning SBERT finished Ô£à")

    train_product_embeddings = generate_reduced_sbert_embeddings(
        96, train_df["title"], bootstrap_identifier
    )
    test_product_embeddings = generate_reduced_sbert_embeddings(
        96, test_df["title"], bootstrap_identifier
    )
    print("Reduced sentence embeddings generated Ô£à")

    print("Training catboost model...")
    # Import and train the catboost classifier here
    catboost_model = train_catboost_classifier(train_product_embeddings, train_df)

    print("Catboost model estimated Ô£à")

    # Random procedure ofcourse values will differ?
    trial_candidates = range(1, 32)
    plane_candidates = range(6, 96)

    """
    run_experiment(
        train_df,
        train_product_embeddings,
        catboost_model,
        trial_candidates,
        plane_candidates,
        total_train_duplicates,
        1,  
    )
    """
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
            "f1_score",
            "comparisons_fraction",
        ],
    )

    df.to_csv(r"../results/bootstrap-runs.csv", index=None)
