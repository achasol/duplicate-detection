"""
This file contains the code to tune a SBERT model using the sentence
transformers library. We tune the model to generate embeddings 
which are very suitable for near duplicate detection. 
"""
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA
import torch
import os

def tune_sbert_model(run_identifier, train_data):

    #The tuned model is already stored 
    if os.path.isdir(f"../models/fine-tuned-model-{run_identifier}"):
        return 

    print("Torch version:", torch.__version__)
    print("Is CUDA enabled?", torch.cuda.is_available())

    # TODO consider taking a random sample of the training data instead of all combinations
    # See impact on performance currently only using 25% of total training data (increase tuning speed)
    # INCREASE TO FULL SAMPLE! (has large impact on quality)
    #chosen_samples = np.random.choice(
    #    range(len(train_data)), size=(len(train_data))
    #)

    train_examples = []
    for product1_index, product2_index in combinations(range(len(train_data)), 2):
        product1 = train_data.iloc[product1_index]
        product2 = train_data.iloc[product2_index]
        train_examples.append(
            InputExample(
                texts=[product1.title, product2.title],
                label=int(product1.id == product2.id),
            )
        )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id, device="cuda")

    train_loss = losses.SoftmaxLoss(
        model=model, sentence_embedding_dimension=384, num_labels=2
    )
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)  # Initial 10

    model.save(f"../models/fine-tuned-model-{run_identifier}")


# Compute PCA on the train embeddings matrix
def generate_reduced_sbert_embeddings(
    desired_dimension, product_titles, run_identifier
):
    model = SentenceTransformer(f"../models/fine-tuned-model-{run_identifier}")
    embeddings = model.encode(product_titles.tolist())
    pca = PCA(n_components=desired_dimension)
    pca.fit(embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=desired_dimension,
        bias=False,
        activation_function=torch.nn.Identity(),
    )

    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module("dense", dense)
    return model.encode(product_titles.tolist())