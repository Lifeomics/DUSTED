# DUSTED: Dual-attention Enhanced Spatial Transcriptomics Denoiser

## Introduction

Spatial transcriptomics is a powerful technique for profiling gene expression in tissue sections while retaining spatial context. However, the data can be noisy, requiring robust methods for denoising. DUSTED (Dual-attention Enhanced Spatial Transcriptomics Denoiser) introduces a sophisticated denoising approach that leverages both gene expression matrices and neighborhood graphs constructed using spatial information.
<div align="center">
  <img src="./resource/model.png" alt="workflow">
  <figcaption>Fig.1 The framework of DUSTED.</figcaption>
</div>

DUSTED's primary function is to estimate clean gene expression levels by incorporating spatial information, which enhances the accuracy and reliability of the data. This process has been proven to improve performance in various downstream tasks, such as gene expression analysis and spatial pattern identification.

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