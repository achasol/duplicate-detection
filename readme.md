# Duplicate detection

Improvements:

1. Exclude comparisons between different brands XX (Poor implementation brand is actually given property for some shops!)

1) Improve on LSH method (try get similar scores to paper)
2) Develop good classifier to swap out golden labels from detector method

Currently pair quality is very bad
Causes:

-Too many comparisons within buckets -> caused by hashing of too many keys to same bucket

-Cause is poor embeddings of model words!
