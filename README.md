# 3D Skeletal Biomechanics Data Extraction Pipeline

## Overview
This repository contains the data preprocessing and extraction architecture for a computer vision-based exercise recognition system. The pipeline is designed to process raw, two-dimensional exercise videos and convert them into lightweight, machine-readable 3D skeletal coordinate sequences using Google's MediaPipe Pose framework. 

## System Requirements
* **Python Environment:** Python 3.11 or higher is strictly required to ensure compatibility with the data processing libraries.
* **Execution Environment:** The extraction and augmentation scripts are designed to be executed interactively within **Jupyter Notebooks** (`.ipynb`). 

## Setup Instructions
1. Clone this repository to your local workspace.
2. Initialize and activate a virtual environment running Python 3.11.
3. Install all required dependencies from the provided text file:
   ```bash
   pip install -r requirements.txt
   ```
4. Register the virtual environment with Jupyter to ensure the notebooks utilize the correct dependencies:
   ```bash
   python -m ipykernel install --user --name=biomechanics-env
   ```

## Data Storage and Architecture
Due to the substantial storage requirements of the raw video files (`.mp4`, `.avi`), the source datasets are not hosted within this repository. 

* **External Hosting:** All raw video data must be downloaded from the designated external Drive link: `https://drive.google.com/drive/folders/1NavekTfKLuthni_oW3AJByvCLCEBf5E5?usp=sharing`

* **Directory Structure:** Upon downloading, the raw videos must be placed into the `data/raw/` directory at the root level of this project. The extraction scripts expect the following structural hierarchy:
  * `data/raw/squat/`
  * `data/raw/push_up/`
  * `data/raw/pull_up/`
  * `data/raw/bench_press/`

## Processing Pipeline
The Jupyter Notebooks execute a linear sequence of operations to transform raw pixels into a balanced, training-ready mathematical dataset.

### 1. 3D Landmark Extraction
The system iterates through the raw video data frame-by-frame, utilizing MediaPipe Pose (set to Complexity Level 2 for maximum accuracy). It isolates 33 distinct anatomical landmarks and extracts their `x`, `y`, and `z` spatial coordinates.

### 2. Data Serialization
Extracted coordinate arrays are serialized and exported as standard `.json` files into the `data/processed/` directory. This isolates the downstream machine learning models from the heavy computational load of processing video pixels.

### 3. Data Integrity and Leakage Prevention
The pipeline rigorously controls the dataset composition to prevent data leakage during model training. It merges real-world datasets (Abdillah, RepCount) with high-quality synthetic avatar data (Riccio's InfiniteRep) while systematically excluding overlapping real-world subsets from the Riccio dataset. 

### 4. Class Balancing and Augmentation
To prevent class bias in the final neural network, the processed data undergoes automated augmentation to reach a strict baseline target (e.g., 250 instances per exercise class). Augmentations are applied directly to the numerical JSON data, bypassing the need for computationally expensive video rendering. These methods include:
* **Spatial Mirroring:** Inverting coordinates along the X-axis to generate symmetric biomechanical data.
* **Temporal Subsampling:** Dropping specific frames to simulate accelerated exercise execution.
* **Gaussian Noise Injection:** Applying randomized numerical variance to spatial coordinates to simulate and train against real-world camera tracking errors.