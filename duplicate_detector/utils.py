import json
import pandas as pd
import math
import numpy as np
from tabulate import tabulate
from sklearn.metrics import f1_score
from .brands import get_brands
import re


def process_results(all_results):
    """
    Process and save the results of multiple bootstrap runs.

    Parameters:
    - all_results (list): List of results from multiple bootstrap runs.
    """
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
    df.to_csv(r"./results/bootstrap-runs.csv", index=None)


def load_dataset():
    """
    Load and preprocess the TV product dataset.

    Returns:
    pandas.DataFrame: Processed DataFrame containing TV product information.
    """
    # Load data from JSON file
    file_path = r"./data/TVs-all-merged.json"
    with open(file_path, "r") as file:
        products = json.load(file)

    minimal_products = []

    dup_identifier = 0

    for product_id in products:
        for product in products[product_id]:
            # Extract relevant information and create a minimal product entry
            minimal_products.append(
                {
                    "shop": product["shop"],
                    "title": product["title"]
                    + "|".join(product["featuresMap"].values()),
                    "id": product["modelID"],
                    "dup_identifier": dup_identifier,
                }
            )
        dup_identifier += 1

    # Create DataFrame from the minimal product information
    minimal_product_df = pd.DataFrame(
        minimal_products, columns=["shop", "title", "id", "dup_identifier"]
    )

    # Extract and process additional features from the 'title' column
    cols = minimal_product_df["title"].str.extract(
        r"([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
    )
    cols = cols.fillna("  ")

    minimal_product_df["brand"] = find_brands(minimal_product_df["title"])

    minimal_product_df["title"] = minimal_product_df["title"].str.lower()

    minimal_product_df["resolution"] = minimal_product_df["title"].str.extract(
        r"\b(\d{3,4})p\b"
    )
    minimal_product_df["refresh_rate"] = minimal_product_df["title"].str.extract(
        r"(\d+)(?=[Hh][Zz])"
    )

    dimensions = minimal_product_df["title"].str.extract(
        r"(\d{2,4})(?:mm|cm) (?:x|X) (\d{2,4})(?:mm|cm)"
    )

    minimal_product_df["dimensions"] = dimensions.fillna("").apply(
        lambda row: "x".join(str(e) for e in row), axis=1
    )

    weights = (
        minimal_product_df["title"]
        .str.extractall(r"(\d+(?:\.\d+)?)\s*lbs")
        .groupby(level=0)[0]
        .apply(list)
    )

    minimal_product_df["weight"] = weights.apply(
        lambda x: min(map(float, x), default=None)
    )

    minimal_product_df["dense_title"] = cols.apply(
        lambda row: "".join(str(e) for e in row), axis=1
    )
    minimal_product_df = minimal_product_df.fillna(" ")
    minimal_product_df["title"] = minimal_product_df[
        ["brand", "resolution", "refresh_rate", "weight", "dimensions"]
    ].apply(lambda row: " ".join(map(str, row)), axis=1)

    return minimal_product_df


def identify_brand(product_title, brand_list):
    """
    Identify the brand in a product title from a list of possible brands.

    Parameters:
    - product_title (str): The product title to be analyzed.
    - brand_list (list): List of possible brands.

    Returns:
    str or None: The identified brand or None if no match is found.
    """
    for brand in brand_list:
        # Create a case-insensitive regular expression pattern for each brand
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)

        # Check if the pattern matches the product title
        if re.search(pattern, product_title):
            return brand

    # Return None if no match is found
    return None


def find_brands(product_titles):
    """
    Identify brands in a list of product titles.

    Parameters:
    - product_titles (list): List of product titles.

    Returns:
    list: List of identified brands for each product title.
    """
    brands_to_identify = get_brands()
    counter = 0
    identified_brands = []

    for title in product_titles:
        identified_brand = identify_brand(title, brands_to_identify)
        if identified_brand is None:
            counter += 1
            identified_brand = f"NAIM{counter}"

        identified_brands.append(identified_brand)

    return identified_brands


def num_duplicates(numbers, df):
    """
    Count the true number of duplicates present in a bootstrap sample.

    Parameters:
    - numbers (list): List of indices corresponding to the bootstrap sample.
    - df (pandas.DataFrame): DataFrame containing product information.

    Returns:
    int: Number of true duplicates in the bootstrap sample.
    """
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
    return duplicates


def summary(
    duplicates_found,
    total_duplicates,
    comparisons_made,
    df_size,
    predictions,
    true_labels,
    print_output=True,
):
    """
    Display the output of the duplicate detection in a suitable way.

    Parameters:
    - duplicates_found (int): Number of duplicates found.
    - total_duplicates (int): Total number of true duplicates in the dataset.
    - comparisons_made (int): Total number of comparisons made.
    - df_size (int): Size of the DataFrame.
    - predictions (list): List of duplicate predictions.
    - true_labels (list): List of true duplicate labels.
    - print_output (bool): Whether to print the output.

    Returns:
    tuple: Tuple of pair quality, pair completeness, F1*-score, F1-score, and fraction comparisons.
    """
    if comparisons_made == 0 or total_duplicates == 0:
        print("No duplicates found")
        return 0, 0, 0, 0, 0

    pair_quality = duplicates_found / comparisons_made
    pair_completeness = duplicates_found / total_duplicates
    f1_star_score = (
        2 * pair_quality * pair_completeness / (pair_quality + pair_completeness)
        if pair_completeness + pair_quality > 0
        else 0
    )
    f1score = f1_score(true_labels, predictions)

    fraction_comparisons = round(comparisons_made / math.comb(df_size, 2), 12)

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
        f1score,
        fraction_comparisons,
    )


def bootstrap_sample(df):
    """
    Create a bootstrap sample from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing product information.

    Returns:
    tuple: Tuple containing train DataFrame, test DataFrame, and number of duplicates in train and test.
    """
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
    """
    Visualize the results of a duplicate detection experiment.

    Parameters:
    - results (list): List of results for each combination of bands and rows.
    """
    print(
        tabulate(
            results,
            headers=[
                "run_identifier",
                "num_bands",
                "num_rows",
                "pair quality",
                "pair completeness",
                "f1*-score",
                "f1score",
                "fraction comparisons",
            ],
            tablefmt="pretty",
        )
    )
