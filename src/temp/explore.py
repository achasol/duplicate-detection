import re
import json
from utils import find_brands
import pandas as pd

import json

"""
brand
refresh_rate
resolution
weight 


"""

uniform_properties = ["brand", "resolution", "refresh_rate", "weight"]


def load_dataset_v2():
    file = open(r"../data/TVs-all-merged.json")
    products = json.load(file)

    minimal_products = []

    dup_identifier = 0

    for product_id in products:
        for product in products[product_id]:
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

    minimal_product_df = pd.DataFrame(
        minimal_products, columns=["shop", "title", "id", "dup_identifier"]
    )

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

    return minimal_product_df["title"]


print(load_dataset_v2())
