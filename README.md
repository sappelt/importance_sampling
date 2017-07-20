# Importance Sampling

This is the first try to implement Stream Clustering: Efficient Kernel-based
Approximation using Importance Sampling by Radha Chitta et al.

Currently working:

- Build up the kernel matrix
- Sample points based on statistical leverage scores
- Compare importance sampling with Bernoulli sampling as proposed in the paper

What is missing:

- Use fast rank-one updates
- Cluster data using approximate kernel clusters
- Finding closest clusters
- Deleting stale clusters