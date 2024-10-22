# My Thesis: Identifying inferences from textual data using deep learning techniques

## Project Overview

This repository contains all the code and research work conducted as part of my thesis on **Natural Language Inference (NLI)**. The goal was to evaluate various deep learning models on multiple NLI datasets and apply ensemble techniques to further improve prediction accuracy. The project also includes code for explainability, fine-tuning for sentiment analysis, and pretraining BERT models, focusing on model evaluation and the application of ensemble techniques to optimize performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Ensemble Technique](#ensemble-technique)
- [Results](#results)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Folder Structure

The repository is organized into the following key folders:

```bash
My_Thesis/
│
├── Explainability/                # Code and experiments related to model explainability
├── Fine_Tuning_For_Sentiment_Analysis/  # Fine-tuning pre-trained models on sentiment analysis tasks
│   ├── Data/ 
├── NLI/                           # Natural Language Inference (NLI) experiments and results
│   ├── Hybrid_Ensemble/           # Hybrid ensemble implementation
│   ├── Models/                    # Pre-trained models used for various datasets
│   ├── Recall_F1-score/           # Scripts to calculate recall and F1 scores
│   ├── Snapshot_ensemble/         # Snapshot ensemble implementation
├── coding_Tranformers/            # Experiments with transformers and model code
├── pretrain_BERT/                 # Pretraining BERT-based models
└── README.md                      # Project README
```



### NLI Folder Structure

- **Hybrid_Ensemble**: Contains the code for using my hybrid ensemble models on key datasets.
- **Models**: Includes the configurations and pretrained models for NLI tasks (SNLI, MNLI, ANLI).
- **Recall_F1-score**: Scripts for calculating recall and F1 scores for the evaluated models.
- **Snapshot_ensemble**: Contains snapshot ensemble implementations for NLI tasks.

## Prerequisites

To run the code in this repository, you'll need:

- Python 3.7 or later
- Key libraries:
  - `transformers`
  - `torch`
  - `sklearn`
  - `numpy`
  - `matplotlib`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ssoulis/My_Thesis.git
```
2. Navigate to the repository:
```bash
cd My_Thesis
```
3. Install all dependencies:
``` bash
pip install -r requirements.txt
```

## Usage
Once the repository is set up, you can run the scripts in the various folders to evaluate models, apply ensemble techniques, or pretrain models. For example, to evaluate the ALBERT-XXLarge model on the SNLI dataset, run:
``` bash
jupyter notebook NLI/albert-xxlarge-v2-snli.ipynb
```
For running the stacking ensemble model with CapsuleNet, use:
``` bash
jupyter notebook NLI/ensemble-CapsuleNet-feuture.ipynb
```
## Model Evaluation




