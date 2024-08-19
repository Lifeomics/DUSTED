# DUSTED: Dual-attention Enhanced Spatial Transcriptomics Denoiser

## Introduction

<div style="text-align: center;">
  <img src="./resource/overview.png" alt="workflow" style="width: 80%;">
  <br>
  <figcaption>Fig.1 Overview of DUSTED denoising procedure.</figcaption>
</div>


Spatial transcriptomics is a powerful technique for profiling gene expression in tissue sections while retaining spatial context. However, the data can be noisy, requiring robust methods for denoising. **DUSTED (Dual-attention Enhanced Spatial Transcriptomics Denoiser)** introduces a sophisticated denoising approach that leverages both gene expression matrices and neighborhood graphs constructed using spatial information.

<div align="center">
  <img src="./resource/model.png" alt="workflow">
  <figcaption>Fig.2 The framework of DUSTED.</figcaption>
</div>

DUSTED enhances the estimation of clean gene expression levels by incorporating spatial information, improving performance in tasks like gene expression analysis and spatial pattern identification. Using a **dual-attention mechanism**, DUSTED focuses on both spatial features and noise variations, interpolating gene expression at any location based on neighboring spots. Additionally, it refines SRT counts to better align with biologically realistic distributions. By leveraging prior biological knowledge, DUSTED accurately fits true gene expression profiles, achieving superior self-supervised SRT data denoising without external auxiliary information.



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