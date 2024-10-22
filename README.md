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
├── Explainability/                # Applied explainability techniques like SHAP and LIME on key models
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
Note that for the scripts to run you need to have the datasets installed. I propose kaggle which is the website I used for my experiments.

## Model Evaluation

The models were evaluated on various NLI datasets, including:

- SNLI (Stanford Natural Language Inference): Dataset for sentence pairs with entailment, contradiction, or neutral labels.
- MNLI (Multi-Genre Natural Language Inference): Dataset for testing NLI models across multiple genres.
- ANLI (Adversarial Natural Language Inference): A challenging dataset for evaluating models on adversarial samples.
- Combined Tasks: A combination of the previous datasets

We evaluated model performance using metrics like accuracy, recall, precision, and F1-score.


## Ensemble Techniques

In this project, we employed three types of ensemble learning techniques:  **Stacking Ensemble Technique**, **Hybrid Ensemble** and **Snapshot Ensemble**. These techniques combine multiple models to improve performance beyond what a single model can achieve.

### 1. Stacking Ensemble Technique

The **stacking ensemble** method is an approach where multiple base models make predictions, and a meta-model is trained to aggregate these predictions into the final output. In our setup, we used models such as ALBERT and DeBERTa as base models, and then fed their predictions into a meta-model.<br />

The following image shows the stacking ensemble technique architecture<br />
![alt text](https://github.com/ssoulis/My_Thesis/blob/main/NLI/Stacking.PNG)

### 2. Hybrid Ensemble Technique
The Hybrid Ensemble approach combines various ensembling methods, including Bayesian combination, majority voting, and stacking, to aggregate predictions from different models.<br />

The following image shows the hybrid ensemble technique architecture that leverages Stacking, Bayesian Combination and majority/weighted vote<br />

![alt text](https://github.com/ssoulis/My_Thesis/blob/main/NLI/SBM-SBW.png)

The following image shows the hybrid ensemble of the 2 Level Stacking model<br />

![alt text](https://github.com/ssoulis/My_Thesis/blob/main/NLI/2LevelStacking.PNG)

### 3. Snapshot Ensemble Technique

The **snapshot ensemble** technique takes a different approach by using **cyclic learning rates** during the training of a single model. Instead of training multiple different models, we save several versions (snapshots) of the same model at different points during training. Each of these snapshots can make predictions, and the final prediction is made by averaging the outputs from all snapshots.

- **Cyclic Learning Rates**: By periodically adjusting the learning rate during training, the model explores different regions of the parameter space, capturing multiple diverse solutions.
- **Model Snapshots**: Several snapshots of the model were saved at different points in the training cycle.
- **Averaging**: The final output is obtained by averaging the predictions from the different model snapshots.

The snapshot ensemble technique is efficient because it doesn’t require training multiple models from scratch. Instead, it reuses different versions of the same model to create an ensemble, thereby reducing the computational cost while still improving performance.

### Results from the Ensemble Techniques

- **Hybrid Ensemble**: The stacking ensemble with CapsuleNet as the meta-model provided significant improvements in terms of F1-score and accuracy across various datasets, outperforming the base models used individually.
- **Snapshot Ensemble**: This technique provided performance gains by using diverse snapshots of the same model, achieving more robust predictions without the need to train multiple models from scratch.





