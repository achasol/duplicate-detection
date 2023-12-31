# Scalable Duplicate Detection

This repo contains my implementation for scalable detection method of (near) duplicate webshop products.
It is part of the course Computer science and Business Analytics at the Erasmus University Rotterdam.

The objective of this implementation is to find near duplicate products in a dataset
without performing an exhaustive comparison of all pairs. Hence we are instructed
to use a locality sensitive hashing (LSH) method to reduce the number of comparisons
in an effective way.

The key aspect of my approach is that, I use the json data of 1624 tv products to create uniform representations.
In essence this boils down to extracting key features of the products
using a variety of regex queries and then joining these features together.
This yields a more dense and uniform representation of the product titles.

I then encode these representations using a one-hot-encoding based on
n-grams of the titles. These one-hot-encoded dense titles are
subsequently used as inputs to a minhashing implementation of
LSH.

For those products hashed to the same bucket I perform a comparison using
their jaccard similarity to evaluate whether the products are duplicates.

## Running the experiment

To run the experiment make sure you install all dependencies and then run:

```
python run.py
```

## Running the test suite

To make sure the experiment runs properly on your machine consider running

```
pytest
```

## Structure of the code

The source code is structured into several files and contained within the duplicate_detector module.

1.  The entrypoint of the method is the main.py file.
2.  The minhash.py file contains a minhashing implementation
3.  The lsh.py file contains the logic required to use minhashing to perform LSH and also keep track of metrics during
    an experiment.
4.  The utils.py file contains a number of utility functions including the function responsible for creating the dense title representations.
5.  The plots.py file uses the results from multiple bootstrap runs of the experiment to easily generate plots on key metrics.

6.  The brands.py file is simply a container containing a python dict with
    all major tv brands extracted from wikipedia.

Automated tests for key functions can be found in the duplicate_detector/tests folder.
