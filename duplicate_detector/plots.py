"""
This file contains the plotting functionality used 
to plot the results from the bootstrap experiments. 

"""


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="white", palette="tab10")

data = pd.read_csv(r"../results/bootstrap-runs-50-simhash-90.csv")

# TODO take average over bootstraps here

# BIN observations based on fraction of comparisons
size = 0.05
# Define bin edges
bin_edges = [
    i * size
    for i in range(
        int(data["comparisons_fraction"].min() / size),
        int(data["comparisons_fraction"].max() / size) + 2,
    )
]

# Create bins and add a new column 'bin' to the DataFrame
data["bin"] = pd.cut(data["comparisons_fraction"], bins=bin_edges, right=False)

# Group by the 'bin' column and calculate some summary statistics
grouped_df = data.groupby("bin").agg(
    {
        "f1_score": ["mean", "std"],
        "f1*_score": ["mean", "std"],
        "pair_quality": ["mean", "std"],
        "pair_completeness": ["mean", "std"],
        "comparisons_fraction": ["mean", "std"],
    }
)


grouped_df = grouped_df.dropna()

print(grouped_df.to_string())

# Create a single figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot F1-measure
axs[0, 0].plot(grouped_df["comparisons_fraction"], grouped_df["f1_score"])
axs[0, 0].set_xlabel("Fraction of comparisons")
axs[0, 0].set_ylabel("F1-measure")
axs[0, 0].set_title("F1-measure")

# Plot F1*-measure
axs[0, 1].plot(grouped_df["comparisons_fraction"], grouped_df["f1*_score"])
axs[0, 1].set_xlabel("Fraction of comparisons")
axs[0, 1].set_ylabel("F1*-measure")
axs[0, 1].set_title("F1*-measure")

# Plot Pair quality
axs[1, 0].plot(grouped_df["comparisons_fraction"], grouped_df["pair_quality"])
axs[1, 0].set_xlabel("Fraction of comparisons")
axs[1, 0].set_ylabel("Pair quality")
axs[1, 0].set_title("Pair quality")

# Plot Pair completeness
axs[1, 1].plot(grouped_df["comparisons_fraction"], grouped_df["pair_completeness"])
axs[1, 1].set_xlabel("Fraction of comparisons")
axs[1, 1].set_ylabel("Pair completeness")
axs[1, 1].set_title("Pair completeness")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
