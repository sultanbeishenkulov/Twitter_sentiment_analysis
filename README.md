# Twitter Sentiment Analysis

## Overview
This project conducts sentiment analysis on Twitter data at an entity level. Given messages and entities, the goal is to categorize the sentiment of the message about the entity into Positive, Negative, or Neutral. Irrelevant messages are considered Neutral.

## Usage
1. **Run the Jupyter Notebook `twitter_sentiment_analysis.ipynb`.**
2. **Follow the instructions provided in the notebook to preprocess data, visualize sentiment distribution, build a model, and evaluate its performance.**
3. **Explore the code and modify it according to your requirements.**

## Data
The dataset consists of Twitter messages (`Tweet`) associated with specific entities (`branch`) and their sentiments (`Sentiment`). The available sentiments are Positive, Negative, and Neutral.

## Preprocessing
- The dataset is cleaned by removing unnecessary columns, dropping missing values, and eliminating duplicates.
- Text data is normalized, tokenized, and cleaned by removing HTML tags, URLs, numbers, punctuation, stopwords, and emojis.

## Data Visualization
- Sentiment distribution is visualized using pie and bar plots.
- A heatmap and statistical analysis are provided to analyze sentiment distribution across different branches.

## Model Building
- Text data is vectorized using TF-IDF.
- A Random Forest classifier is trained on the TF-IDF features.
- Model performance is evaluated using accuracy, confusion matrix, and classification report.

## Dependencies
- Python 3
- pandas
- numpy
- seaborn
- matplotlib
- nltk
- scikit-learn
- joblib





