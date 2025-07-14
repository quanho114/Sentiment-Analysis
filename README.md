# Sentiment Analysis on Amazon Fine Food Reviews

This project performs sentiment analysis on Amazon product reviews using two different machine learning approaches:

- A traditional Naive Bayes classifier with TF-IDF features
- A fine-tuned BERT model using Hugging Face Transformers

The objective is to classify reviews into three sentiment classes: Positive, Neutral, and Negative.

## Problem Statement

Given a product review (text), the task is to determine its sentiment. This is a multi-class classification problem, where each review is categorized as:

- Positive
- Neutral
- Negative

This problem has practical applications in e-commerce, customer support, and brand monitoring.

## Dataset

- Dataset: Amazon Fine Food Reviews
- Source: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- Samples: 568,454 product reviews

Key fields:
- `Text`: Full text of the review
- `Score`: Star rating (1 to 5), used for labeling sentiment
- `Summary`, `ProductId`, `UserId`, etc.

Label conversion:
- Score 1–2: Negative
- Score 3: Neutral
- Score 4–5: Positive

After downloading, place the CSV file inside the `data/` directory.

## Methodology

### Preprocessing
- Remove punctuation and special characters
- Convert text to lowercase
- Remove stopwords
- Map review scores to sentiment labels

### Naive Bayes Classifier
- Convert text to TF-IDF vectors
- Train using Multinomial Naive Bayes from scikit-learn
- Evaluate using precision, recall, and F1-score

### BERT Fine-Tuning
- Use pretrained `bert-base-uncased` from Hugging Face
- Tokenize and generate attention masks
- Fine-tune using PyTorch and Transformers library
- Evaluate using the same metrics

## Results

### Naive Bayes Classification Report

| Sentiment | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.96      | 0.07   | 0.13     | 12,301  |
| Neutral  | 0.89      | 0.00   | 0.00     | 6,396   |
| Positive | 0.79      | 1.00   | 0.88     | 66,564  |
| Accuracy |           |        | 0.79     | 85,261  |

Observation: The Naive Bayes model struggles with class imbalance and tends to predict the Positive class most often.

### BERT Classification Report

| Sentiment | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.81      | 0.84   | 0.83     | 16,401  |
| Positive | 0.44      | 0.73   | 0.55     | 8,528   |
| Neutral  | 0.98      | 0.91   | 0.95     | 88,752  |
| Accuracy |           |        | 0.89     | 113,681 |

Observation: BERT performs significantly better, especially on the Neutral and Negative classes, showing its ability to handle contextual language understanding.

## How to Run

1. Clone the repository
git clone https://github.com/quanho114/Sentiment-Analysis.git
cd Sentiment-Analysis

3. Run the notebooks
