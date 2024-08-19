# Network Communities and Bayesian Classification

This repository contains implementations for detecting network communities using the Radicchi et al. algorithm and a Naive Bayes classifier for binary classification tasks. The focus is on practical applications of network analysis and machine learning techniques.

## Description

### Network Communities

This section involves the implementation of the Radicchi et al. algorithm to identify and classify communities within a given network. Key features include:

- **Edge-Clustering Coefficient Calculation**: Compute the edge-clustering coefficient to understand the local clustering around edges in the network.
- **Network Decomposition**: Decompose the network by iteratively removing edges with the lowest edge-clustering coefficient and track the formation of communities.
- **Community Classification**: Classify the resulting subgraphs into strong and weak communities based on their internal and external connections.
- **Visualization**: Generate visual representations of the network and its community structure, including dendrograms that depict the hierarchical clustering.

### Naive Bayes Classifier

This section focuses on implementing a Naive Bayes classifier to predict binary outcomes based on a set of features. Key features include:

- **Model Training**: Train the Naive Bayes classifier using a dataset where each feature represents a categorical variable.
- **Feature Importance**: Identify the most significant features contributing to the classification by analyzing the log-likelihood ratios.
- **Classification**: Classify new samples based on the trained model and evaluate the model’s performance using accuracy metrics.
- **Application to Test Data**: Apply the classifier to test datasets and analyze its performance across different scenarios.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following libraries:

- `numpy`
- `pandas`
- `networkx`
- `matplotlib`

You can install these dependencies via pip:

```bash
pip install numpy pandas networkx matplotlib
