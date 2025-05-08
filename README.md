# DUSTED: Dual-attention Enhanced Spatial Transcriptomics Denoiser

## Introduction

<div align="center">
  <img src="./resource/overview.png" alt="workflow" style="width: 100%;">
  <figcaption>Fig.1 Overview of DUSTED denoising procedure.</figcaption>
</div>


Spatial transcriptomics is a powerful technique for profiling gene expression in tissue sections while retaining spatial context. However, the data can be noisy, requiring robust methods for denoising. **DUSTED (Dual-attention Enhanced Spatial Transcriptomics Denoiser)** introduces a sophisticated denoising approach that leverages both gene expression matrices and neighborhood graphs constructed using spatial information.

<div align="center">
  <img src="./resource/model.png" alt="workflow">
  <figcaption>Fig.2 The framework of DUSTED.</figcaption>
</div>

DUSTED enhances the estimation of clean gene expression levels by incorporating spatial information, improving performance in tasks like gene expression analysis and spatial pattern identification. Using a **dual-attention mechanism**, DUSTED focuses on both spatial features and noise variations, interpolating gene expression at any location based on neighboring spots. Additionally, it refines SRT counts to better align with biologically realistic distributions. By leveraging prior biological knowledge, DUSTED accurately fits true gene expression profiles, achieving superior self-supervised SRT data denoising without external auxiliary information.

# Update on Evaluation Metrics

We have updated(05/08/2025) our evaluation methodology for dimensionality reduction and clustering. Initially, we used STAGATE for dimensionality reduction followed by the **mclust** clustering method. However, after further consideration, we realized that PCA (Principal Component Analysis) is a more suitable choice for dimensionality reduction. Therefore, we have updated the corresponding evaluation results using **PCA for dimensionality reduction** followed by **mclust** clustering. The updated ARI, NMI, and HS results are as follows:

### ARI (Adjusted Rand Index) Results

| Sample  | Raw        | STAGATE    | DCA        | MAGIC      | Smoother   | DUSTED     | stlearn    | sprod      |
|---------|------------|------------|------------|------------|------------|------------|------------|------------|
| 151676  | 0.31       | 0.49       | 0.23       | 0.21       | 0.31       | 0.52       | 0.45       | 0.37       |
| 151675  | 0.27       | 0.36       | 0.31       | 0.26       | 0.30       | 0.56       | 0.49       | 0.33       |
| 151674  | 0.37       | 0.48       | 0.23       | 0.33       | 0.38       | 0.46       | 0.54       | 0.29       |
| 151673  | 0.51       | 0.45       | 0.36       | 0.27       | 0.48       | 0.57       | 0.55       | 0.37       |
| 151672  | 0.40       | 0.54       | 0.15       | 0.46       | 0.47       | 0.57       | 0.53       | 0.45       |
| 151671  | 0.42       | 0.51       | 0.09       | 0.39       | 0.44       | 0.50       | 0.48       | 0.39       |
| 151670  | 0.20       | 0.32       | 0.18       | 0.17       | 0.23       | 0.29       | 0.26       | 0.27       |
| 151669  | 0.31       | 0.35       | 0.01       | 0.18       | 0.21       | 0.47       | 0.30       | 0.33       |
| 151510  | 0.36       | 0.41       | 0.23       | 0.23       | 0.18       | 0.49       | 0.36       | 0.17       |
| 151509  | 0.23       | 0.53       | 0.21       | 0.26       | 0.29       | 0.45       | 0.45       | 0.19       |
| 151508  | 0.30       | 0.49       | 0.22       | 0.24       | 0.24       | 0.45       | 0.47       | 0.32       |
| 151507  | 0.42       | 0.54       | 0.31       | 0.22       | 0.20       | 0.46       | 0.40       | 0.22       |
| **Mean**| **0.3417** | **0.4558** | **0.2108** | **0.2683** | **0.3108** | **0.4825** | **0.4400** | **0.3083** |

### NMI (Normalized Mutual Information) Results

| Sample  | Raw      | STAGATE  | DCA      | MAGIC    | Smoother | DUSTED   | stlearn  | sprod    |
|---------|----------|----------|----------|----------|----------|----------|----------|----------|
| 157616  | 0.46     | 0.63     | 0.35     | 0.40     | 0.46     | 0.67     | 0.55     | 0.49     |
| 157616  | 0.40     | 0.50     | 0.45     | 0.40     | 0.43     | 0.72     | 0.57     | 0.47     |
| 151674  | 0.50     | 0.58     | 0.29     | 0.43     | 0.46     | 0.63     | 0.65     | 0.44     |
| 151673  | 0.60     | 0.63     | 0.48     | 0.46     | 0.58     | 0.70     | 0.65     | 0.49     |
| 151672  | 0.55     | 0.66     | 0.29     | 0.49     | 0.53     | 0.67     | 0.60     | 0.49     |
| 151671  | 0.55     | 0.65     | 0.21     | 0.50     | 0.51     | 0.65     | 0.59     | 0.48     |
| 151670  | 0.42     | 0.53     | 0.20     | 0.30     | 0.39     | 0.51     | 0.48     | 0.39     |
| 151669  | 0.49     | 0.58     | 0.13     | 0.32     | 0.35     | 0.62     | 0.49     | 0.41     |
| 151510  | 0.47     | 0.61     | 0.29     | 0.34     | 0.31     | 0.62     | 0.51     | 0.32     |
| 151509  | 0.23     | 0.66     | 0.31     | 0.35     | 0.38     | 0.64     | 0.58     | 0.34     |
| 151508  | 0.43     | 0.63     | 0.26     | 0.35     | 0.37     | 0.64     | 0.56     | 0.39     |
| 151507  | 0.53     | 0.66     | 0.42     | 0.37     | 0.38     | 0.65     | 0.57     | 0.39     |
| **Mean**| **0.4692**| **0.6100**| **0.3067**| **0.3925**| **0.4292**| **0.6433**| **0.5667**| **0.4250** |

### HS (Homogeneity Score) Results

| Sample  | Raw    | STAGATE | DCA    | MAGIC  | Smoother | DUSTED | stlearn | sprod  |
|---------|--------|---------|--------|--------|----------|--------|---------|--------|
| 157616  | 0.46   | 0.63    | 0.35   | 0.44   | 0.46     | 0.66   | 0.54    | 0.49   |
| 157616  | 0.40   | 0.50    | 0.45   | 0.41   | 0.43     | 0.71   | 0.56    | 0.47   |
| 151674  | 0.50   | 0.57    | 0.30   | 0.43   | 0.48     | 0.62   | 0.65    | 0.44   |
| 151673  | 0.59   | 0.63    | 0.47   | 0.47   | 0.56     | 0.69   | 0.64    | 0.48   |
| 151672  | 0.49   | 0.60    | 0.25   | 0.46   | 0.48     | 0.61   | 0.54    | 0.41   |
| 151671  | 0.48   | 0.58    | 0.18   | 0.44   | 0.45     | 0.58   | 0.53    | 0.43   |
| 151670  | 0.34   | 0.44    | 0.17   | 0.24   | 0.32     | 0.41   | 0.39    | 0.32   |
| 151669  | 0.41   | 0.49    | 0.11   | 0.27   | 0.30     | 0.53   | 0.41    | 0.35   |
| 151510  | 0.45   | 0.57    | 0.28   | 0.34   | 0.31     | 0.62   | 0.48    | 0.32   |
| 151509  | 0.37   | 0.63    | 0.29   | 0.34   | 0.37     | 0.64   | 0.57    | 0.33   |
| 151508  | 0.42   | 0.61    | 0.25   | 0.36   | 0.37     | 0.64   | 0.56    | 0.38   |
| 151507  | 0.54   | 0.67    | 0.42   | 0.39   | 0.39     | 0.67   | 0.59    | 0.40   |
| **Mean**| **0.4542** | **0.5767** | **0.2933** | **0.3825** | **0.4100** | **0.6150** | **0.5383** | **0.4017** |

---


## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- torch-geometric
- matplotlib
### Setup
1. Clone the repository:
   ```bash
   mkdir DUSTED
   git clone git@github.com:Lifeomics/DUSTED.git
   cd DUSTED
   ```
### Project Structure
 ```
 ├── model.py              # Contains the DUSTED model and other architectures
 ├── loss.py               # Contains the custom loss functions  
 ├── trainer.py            # Script for training the DUSTED model
 ├── README.md             # Project documentation
 └── requirements.txt      # Python dependencies
 ```
## Cite
```
Zhu, J., Li, Y., Tang, Z., & Chang, C. (2025). DUSTED: Dual-Attention Enhanced Spatial Transcriptomics Denoiser. Proceedings of the AAAI Conference on Artificial Intelligence, 39(1), 1219-1227. https://doi.org/10.1609/aaai.v39i1.32110
```
