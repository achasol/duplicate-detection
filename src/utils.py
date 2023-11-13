import json
import pandas as pd
import math
import numpy as np
from tabulate import tabulate
from sklearn.metrics import f1_score 

# Function to load the product dataset and generate a minimal representation
def load_dataset():
    file = open(r"../data/TVs-all-merged.json")
    products = json.load(file)

    minimal_products = []

    for product_id in products:
        for product in products[product_id]:
            minimal_products.append(
                {
                    "shop": product["shop"],
                    "title": product["title"],
                    "id": product["modelID"],
                }
            )
    minimal_product_df = pd.DataFrame(minimal_products, columns=["shop", "title", "id"])

    return minimal_product_df


# Count the true number of duplicates present in a bootstrap sample
def num_duplicates(numbers, df):
    counter = {}
    for number in numbers:
        if df.iloc[number].id in counter:
            counter[df.iloc[number].id][df.iloc[number].shop] = True
        else:
            counter[df.iloc[number].id] = {}
            counter[df.iloc[number].id][df.iloc[number].shop] = True

    duplicates = 0

    for count in list(counter.values()):
        duplicates += math.comb(len(count.keys()), 2)
    # print(duplicates)
    return duplicates


# Method used to display the output of the duplicate detection in a suitable way
def summary(
    duplicates_found, total_duplicates, comparisons_made, df_size,predictions,true_labels, print_output=True
):
    if comparisons_made == 0 or total_duplicates == 0:
        print("No duplicates found")
        return 0, 0, 0, 0

    pair_quality = duplicates_found / comparisons_made
    pair_completeness = duplicates_found / total_duplicates
    f1_star_score = (
        2 * pair_quality * pair_completeness / (pair_quality + pair_completeness)
        if pair_completeness + pair_quality > 0
        else 0
    )
    f1score = f1_score(true_labels,predictions)
    fraction_comparisons = round(comparisons_made / math.comb(df_size, 2), 6)
    if print_output:
        print(f"Total number of duplicates: {total_duplicates}")
        print(f"Duplicates found: {duplicates_found}")
        print(
            f"Comparisons made: {comparisons_made} / {math.comb(df_size,2)} ({fraction_comparisons})"
        )
        print(
            f"Pair Quality: {round(pair_quality,3)} ({duplicates_found} / {comparisons_made})"
        )
        print(
            f"Pair Completeness: {round(pair_completeness,3)} ({duplicates_found} / {total_duplicates})"
        )
        print(f"F1*-score: {round(f1_star_score,6)}")
        print(f"F1-score: {round(f1score,6)}")
        print("#" * 36)
    return (
        pair_quality,
        pair_completeness,
        f1_star_score,
        f1score , 
        fraction_comparisons,
    )


def bootstrap_sample(df):
    train_set = list(
        set(np.random.randint(0, len(df) - 1, size=len(df)))
    )  # Drop non-unique duplicates
    test_set = list(set([i for i in range(len(df))]) - set(train_set))

    df_train = df.iloc[train_set]
    df_test = df.iloc[test_set]

    return (
        df_train,
        df_test,
        num_duplicates(train_set, df),
        num_duplicates(test_set, df),
    )


def visualize_results(results):
    print(
        tabulate(
            results,
            headers=[
                "run_identifier",
                "trial",
                "planes",
                "pair quality",
                "pair completeness",
                "f1*-score",
                "f1score"
                "fraction comparisons",
            ],
            tablefmt="pretty",
        )
    )
