# Word Vectors Visualization Project

In this Project we will explore word vectors.
In natural language processing, we represent each word as a vector consisting of numbers.
The vector encodes the meaning of the word. These numbers (or weights) for each word are learned using various machine
learning models, Here we will see how to use them.

In this we will
- Predict analogies between words.
- Use PCA to reduce the dimensionality of the word embeddings and visualizing them in a 2D space.
- Compare word embeddings by using a similarity measure.
- Understand how these vector space models work.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Details](#project-details)
  - [Data Loading](#data-loading)
  - [Word Vectors](#word-vectors)
    - [Cosine Similarity](#1-2)
    - [Euclidean Distance](#1-3)
    - [Finding the Country of each Capital](#1-4)
    - [Model Accuracy](#1-5)
  - [PCA for Dimensionality Reduction](#PCA_for_Dimensionality_Reduction)
  - [Visualization](#Visualization)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project aims to provide a visual representation of word vectors to understand the relationships between words in a corpus. By using PCA, we reduce the dimensions of the word vectors to visualize them in a 2D plot. Additionally, we explore a method to predict countries from their capital cities using word vectors.

## Requirements

- Python 3.9
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

Clone the repository and download the required data:

```bash
git clone https://github.com/yourusername/word-vectors-visualization.git
cd word-vectors-visualization
download GoogleNews-vectors-negative300.bin
```

## Usage

Run the Jupyter notebook to see the word vectors visualization and how the country prediction works:

```bash
jupyter notebook Word_Vectors.ipynb
```

## Project Details

### Data Loading

The project begins by importing a dataset containing word embeddings for a large vocabulary of words and capital for each city. This dataset is used to calculate correct predictions.

#### Cosine Similarity

Cosine similarity is a metric used to measure how similar two word vectors are. It calculates the cosine of the angle between two vectors, which helps in determining the similarity in their direction.

#### Euclidean Distance

Euclidean distance is another metric used to measure the similarity between word vectors. It calculates the straight-line distance between two points in the vector space, providing a measure of their dissimilarity.

#### Finding the Country of each Capital

To predict the country corresponding to each capital, we will write a function that takes in three words, and the embeddings dictionary. and returns the country for the corresponding capital.

For example, given the following words: 

    1: Athens 2: Greece 3: Baghdad,

    our task is to predict the country 4: Iraq.

The country with the highest similarity is selected as the predicted country.

#### Model Accuracy

To evaluate the model's accuracy, we compare the predicted countries with the actual countries for a set of capital cities. The accuracy is calculated as the percentage of correct predictions out of the total number of predictions.

### PCA for Dimensionality Reduction

PCA (Principal Component Analysis) is applied to reduce the dimensionality of the word vectors from their high-dimensional space to a 2D space. This reduction helps in visualizing the relationships between words in a plot. 

### Visualization

The 2D PCA-transformed word vectors are then plotted using Matplotlib, where each word is represented as a point.

## Results

![Word Vectors Visualization](/output.png)

The visualization shows clusters of words with similar meanings. For example, words like "gas," "oil," and "petroleum" are grouped together, while "sad," "joyful," and "happy" form another cluster.

## Conclusion

This project demonstrates the power of word vectors, word embeddings and PCA in capturing and visualizing the semantic relationships between words. The additional model for predicting countries from capitals showcases practical applications of word vectors in natural language processing.


