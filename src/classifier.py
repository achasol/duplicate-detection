"""
Use a catboost classifier based on similarity measures 

1) Get embeddings from data 
2) Construct dataset with true label and similarity metrics (look for library)
3) Finetune catboost model using optuna (Not really required does not fit in 6 pages) XX
4) See achieved performance 



"""
from utils import load_dataset

import catboost as cb

# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from itertools import combinations

# Load train data
minimal_product_df = load_dataset()[:500]


def gradient_boosting_catboost() -> cb.CatBoostClassifier:
    clf = cb.CatBoostClassifier(
        **{
            "learning_rate": 0.02,
            "iterations": 500,
            "depth": 4,
            "boosting_type": "Plain",
            "l2_leaf_reg": 1e-1,
            "border_count": 128,  # For CPU use 254 (thread blocks)
            # "subsample": trial.suggest_float("cb_subsample", 0.5, 1.0, step=0.1),
            # "colsample_bylevel": 0.4,
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
            "verbose": True,
            "task_type": "GPU",
            "early_stopping_rounds": 100,
        }
    )

    return clf


def generate_catboost_sample(embedding1, embedding2):
    return [*embedding1, *embedding2]


def train_catboost_classifier(embeddings, df):
    labels = []
    samples = []
    for product1_index, product2_index in combinations(range(len(df)), 2):
        product1 = df.iloc[product1_index]
        product2 = df.iloc[product2_index]
        samples.append(
            [
                *embeddings[product1_index],
                *embeddings[product2_index],
            ]
        )
        labels.append(int(product1.id == product2.id))

    # resample the data
    smote = SMOTE(
        sampling_strategy="auto", random_state=42
    )  # You can adjust the sampling_strategy as needed
    samples, labels = smote.fit_resample(samples, labels)

    catboost_model = gradient_boosting_catboost()
    catboost_model.fit(samples, labels)

    return catboost_model
