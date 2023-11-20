from sklearn.feature_extraction.text import CountVectorizer


def one_hot_encode_n_shingles(product_titles, n):
    """
    One-hot encodes n-shingles of multiple sentences.

    Parameters:
    - sentences (list): List of sentences (strings).
    - n (int): Size of n-shingles.

    Returns:
    - one_hot_matrix (numpy array): One-hot encoded matrix of n-shingles for each sentence.
    - feature_names (list): List of feature names corresponding to the columns of the matrix.
    """

    # Use CountVectorizer for one-hot encoding
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(n, n), binary=True)
    one_hot_matrix = vectorizer.fit_transform(product_titles).toarray()

    return one_hot_matrix
