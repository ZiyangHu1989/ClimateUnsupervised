# Project Proposal: Unsupervised Learning for Global Climate Pattern Discovery

## Project Idea

Our project investigates whether meaningful global climate patterns can be discovered using purely **unsupervised learning**. Each geographic location will be represented as a high-dimensional vector describing its long-term monthly climate conditions. We will then apply dimensionality-reduction and clustering algorithms to identify latent climate structures.

The central research question is:

> **Can unsupervised machine-learning methods recover realistic global climate regions without using predefined climate labels during training?**

The model will not use geographic coordinates or Köppen–Geiger climate labels as training features. Latitude and longitude will be retained only for visualization. Köppen–Geiger labels will be introduced only after clustering for external evaluation, so the learning stage remains fully unsupervised.

## Implementation Plan

### 1. Data Collection

We plan to use **ERA5-Land monthly climate data from 1991–2020**, which closely matches the time period of the modern Köppen–Geiger 1991–2020 climate reference.

Each valid land grid cell will be treated as one sample. The initial feature vector will contain:

- 12 monthly mean temperatures
- 12 monthly precipitation values

Thus, each location will be represented as a **24-dimensional vector**:

\[
x_i=[T_{Jan},\ldots,T_{Dec},P_{Jan},\ldots,P_{Dec}]
\]

Latitude and longitude will be stored separately and will **not** be used for training or clustering.

### 2. Preprocessing

The preprocessing pipeline will include:

- removing ocean and invalid grid cells;
- converting or normalizing units where necessary;
- applying `log(1 + P)` to precipitation if its distribution is strongly skewed;
- standardizing all features using z-score normalization;
- checking correlations among the 24 climate features.

### 3. Dimensionality Reduction

We will use **Principal Component Analysis (PCA)** as the main dimensionality-reduction method.

PCA will be used for two purposes:

1. **Clustering representation:** retain enough principal components to explain approximately 90–95% of the total variance.
2. **Interpretation and visualization:** inspect explained variance and component loadings to identify major climate dimensions such as temperature level, wetness, and seasonality.

We may additionally use **UMAP** for 2D nonlinear visualization. UMAP will be treated primarily as a visualization tool rather than the main clustering space.

### 4. Clustering

We will compare two main clustering algorithms:

#### K-means

K-means will serve as the main hard-clustering baseline. We will test a reasonable range of cluster counts, for example:

\[
K=2,\ldots,15
\]

#### Gaussian Mixture Model (GMM)

GMM will provide probabilistic cluster memberships and may better represent gradual transitions between climate regimes.

The number of clusters will be selected **without using Köppen–Geiger labels**.

Possible internal evaluation criteria include:

- Silhouette Score
- Davies–Bouldin Index
- K-means inertia / elbow analysis
- Cluster stability across repeated runs or subsamples
- BIC and AIC for GMM

### 5. Visualization and Interpretation

We will visualize the learned structure in several ways:

- PCA scatter plots colored by cluster assignment;
- optional UMAP 2D visualization;
- global geographic maps showing cluster assignments at each latitude/longitude location;
- average monthly temperature and precipitation profiles for each cluster.

This will allow us to examine whether geographically distant regions with similar climates are assigned to the same cluster, even though geographic coordinates were never used during training.

### 6. Model Freeze

Before using any Köppen–Geiger labels, we will freeze:

- preprocessing choices;
- PCA dimensionality;
- number of clusters;
- K-means and GMM models;
- final cluster assignments.

This step ensures that the external reference does not influence model selection.

### 7. External Evaluation

After the unsupervised pipeline is complete, we will compare the learned climate clusters with the **Köppen–Geiger 1991–2020 climate classification**.

Possible evaluation metrics include:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- V-measure
- cluster-versus-Köppen contingency matrices
- geographic agreement/disagreement maps

The purpose is not to force the clusters to reproduce Köppen–Geiger exactly, but to determine which known climate structures naturally emerge from the data and where the data-driven clustering differs from the traditional rule-based system.

## Group Member Responsibilities

### Fu — Data and Preprocessing

- Download and organize ERA5-Land data.
- Construct the 24-dimensional climate feature matrix.
- Handle missing values, masking, unit conversion, and spatial alignment.
- Implement precipitation transformation and standardization.

### Liang — Dimensionality Reduction and Visualization

- Implement PCA.
- Analyze explained variance and PCA loadings.
- Implement UMAP if needed.
- Produce 2D feature-space visualizations and geographic cluster maps.

### Zhang — Clustering

- Implement K-means and GMM.
- Perform unsupervised selection of the number of clusters.
- Compute internal clustering metrics.
- Analyze cluster stability and cluster climate profiles.

### Zhu — Evaluation and Integration

- Load and align the Köppen–Geiger reference data.
- Implement ARI, NMI, V-measure, and contingency analysis.
- Analyze geographic agreement and disagreement regions.
- Integrate the full workflow and prepare the final report/presentation.

## Expected Outcome

The final system will test whether global climate structure can be recovered from climate observations alone using unsupervised machine learning. The main pipeline is:

```text
ERA5-Land 1991–2020
        ↓
24-D climate vectors
        ↓
Preprocessing
        ↓
PCA
        ↓
K-means / GMM
        ↓
Internal evaluation
        ↓
PCA / UMAP / geographic visualization
        ↓
Freeze model
        ↓
Köppen–Geiger 1991–2020 comparison
        ↓
ARI / NMI / V-measure / disagreement analysis
```

The project will focus on interpretable, reproducible unsupervised learning rather than maximizing agreement with predefined climate labels.
