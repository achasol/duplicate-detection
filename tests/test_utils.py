import pytest
import numpy as np
import pandas as pd

# Assuming the functions are in a module called "duplicate_detection"

from duplicate_detector.utils import (
    identify_brand,
    find_brands,
    num_duplicates,
    summary,
    bootstrap_sample,
    visualize_results,
)


@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {
        "id": [1, 2, 3, 1, 2],
        "shop": ["A", "D", "C", "C", "B"],
        "title": ["Product A", "Product B", "Product C", "Product A", "Product B"],
    }
    return pd.DataFrame(data)


def test_identify_brand():
    brand_list = ["BrandA", "BrandB", "BrandC"]

    # Test case with a matching brand
    product_title_matching = "This is a BrandB product"
    assert identify_brand(product_title_matching, brand_list) == "BrandB"

    # Test case with no matching brand
    product_title_non_matching = "This product has no brand"
    assert identify_brand(product_title_non_matching, brand_list) is None


def test_find_brands(sample_dataframe):
    # Test case with a DataFrame
    product_titles = sample_dataframe["title"].tolist()
    result = find_brands(product_titles)
    assert len(result) == len(product_titles)


def test_num_duplicates(sample_dataframe):
    # Test case with a DataFrame
    numbers = [0, 1, 2, 3, 4]
    result = num_duplicates(numbers, sample_dataframe)
    assert result == 2  # There are two duplicate pairs: (0, 3) and (1, 4)


def test_bootstrap_sample(sample_dataframe):
    # Test case with a DataFrame
    result = bootstrap_sample(sample_dataframe)
    assert (
        len(result) == 4
    )  # Tuple with four elements: df_train, df_test, num_duplicates_train, num_duplicates_test


def test_visualize_results(capsys):
    # Test case with dummy results
    results = [
        [1, 5, 10, 0.8, 0.6, 0.75, 0.7, 0.05],
        [2, 8, 12, 0.9, 0.7, 0.85, 0.8, 0.1],
    ]
    visualize_results(results)

    # Capture the printed output and check if it contains specific strings
    captured = capsys.readouterr()
    assert "run_identifier" in captured.out
    assert "num_bands" in captured.out
    assert "num_rows" in captured.out
    assert "pair quality" in captured.out
    assert "pair completeness" in captured.out
    assert "f1*-score" in captured.out
    assert "f1score" in captured.out
    assert "fraction comparisons" in captured.out
