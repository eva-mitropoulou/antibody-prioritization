# Unsupervised Antibody Sequence Landscape

Clustering used sequence-derived features only. Labels were used after
clustering for enrichment summaries, not for assigning clusters.

| Metric | Value |
|---|---:|
| Rows | 5573 |
| Feature source | cached_pair_embeddings |
| Cluster method | KMeans |
| Number of clusters | 9 |
| Assigned row fraction | 1.0000 |
| Silhouette score | 0.145587061939435 |

## Cluster Sizes

| Cluster | Rows |
|---|---:|
| 4 | 1613 |
| 1 | 1599 |
| 7 | 601 |
| 6 | 477 |
| 8 | 412 |
| 3 | 404 |
| 5 | 178 |
| 2 | 176 |
| 0 | 113 |

## Label Enrichment After Clustering

| Cluster | Rows | Label 0 | Label 1 | Positive fraction among labeled |
|---|---:|---:|---:|---:|
| 0 | 113 | 43 | 70 | 0.6195 |
| 1 | 1599 | 723 | 876 | 0.5478 |
| 2 | 176 | 120 | 56 | 0.3182 |
| 3 | 404 | 181 | 223 | 0.5520 |
| 4 | 1613 | 693 | 920 | 0.5704 |
| 5 | 178 | 86 | 92 | 0.5169 |
| 6 | 477 | 58 | 419 | 0.8784 |
| 7 | 601 | 227 | 374 | 0.6223 |
| 8 | 412 | 161 | 251 | 0.6092 |

## Artifacts

- `reports/unsupervised_antibody_clusters.csv`
- `reports/metrics/unsupervised_antibody_landscape_metrics.json`
- `reports/figures/unsupervised_antibody_landscape.png`
