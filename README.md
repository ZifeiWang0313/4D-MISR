# 4D-MISR: A Unified Model for Low-Dose Super-Resolution Imaging via Feature Fusion

This repository contains the code for the **4D-MISR** framework, a deep learning-based method designed for low-dose super-resolution imaging of beam-sensitive materials using 4D-STEM (Four-Dimensional Scanning Transmission Electron Microscopy) data. The 4D-MISR method integrates multi-image super-resolution (MISR) principles with a dual-stage neural network architecture to enhance image resolution while minimizing radiation damage to samples.

## Overview

Electron microscopy, particularly 4D-STEM, provides high-resolution imaging of materials. However, beam-sensitive materials like biological specimens and certain inorganic materials face challenges due to radiation-induced damage. The 4D-MISR framework addresses this by leveraging a multi-view geometry approach that uses low-dose 4D-STEM datasets to produce high-resolution images while significantly reducing radiation exposure.

4D-MISR utilizes a deep learning pipeline combining feature fusion from multiple low-dose views and physics-informed denoising to reconstruct atomic-scale details in materials. The network employs a dual-branch architecture that first extracts local features from each image view and then refines these using global angular information. 

## Features

- **Low-Dose Imaging**: Achieves high-resolution reconstructions with significantly fewer electrons compared to traditional methods like ptychography.
- **Feature Fusion**: Integrates angular and spatial features using a deep learning-based network for enhanced image quality.
- **Multi-View Geometry**: Utilizes diverse angular perspectives from 4D-STEM data to enhance resolution without compromising contrast.
- **Wide Applicability**: Applicable to various material systems, including crystalline, semi-crystalline, and amorphous materials, under low-dose imaging conditions.

## Installation

### Requirements

- Python 3.x
- PyTorch (>=1.8.0)
- numpy
- scipy
- matplotlib

To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
