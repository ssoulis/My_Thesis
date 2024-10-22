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

My_Thesis/ │ ├── Explainability/ # Code and experiments related to model explainability ├── Fine_Tuning_For_Sentiment_Analysis/ # Fine-tuning pre-trained models on sentiment analysis tasks ├── NLI/ # Natural Language Inference (NLI) experiments and results │ ├── Hybrid_Ensemble/ # Code for stacking ensemble techniques │ ├── Models/ # Pre-trained models used for various datasets │ ├── Recall_F1-score/ # Scripts to calculate recall and F1 scores │ ├── Snapshot_ensemble/ # Snapshot ensemble implementation │ ├── albert-xxlarge-v2-snli.ipynb # Model evaluation on SNLI dataset using ALBERT │ ├── deberta3-MNLI-m.ipynb # Model evaluation on MNLI dataset using DeBERTa │ ├── ensemble-CapsuleNet-feuture.ipynb # Ensemble with CapsuleNet as meta model │ └── [additional NLI evaluations and ensemble codes] ├── coding_Tranformers/ # Experiments with transformers and model code ├── pretrain_BERT/ # Pretraining BERT-based models └── README.md # Project README
