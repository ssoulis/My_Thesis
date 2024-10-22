# My Thesis: Identifying inferences from textual data using deep learning techniques

## Project Overview

This repository contains all the code and research work conducted as part of my thesis on **Natural Language Inference (NLI)**. The goal was to evaluate various deep learning models on multiple NLI datasets and apply ensemble techniques to further improve prediction accuracy. The project also includes code for explainability, fine-tuning for sentiment analysis, and pretraining BERT models, focusing on model evaluation and the application of stacking ensemble techniques to optimize performance.

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

- **Hybrid_Ensemble**: Contains the code for applying stacking ensemble techniques by combining predictions from multiple models.
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

Install all required dependencies using the following command:

```bash
pip install -r requirements.txt
```




