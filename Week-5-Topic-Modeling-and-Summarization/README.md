# Week 5 – Topic Modeling and Extractive Text Summarization

## Project Overview

CivicSift AI is an NLP-based project that analyzes NYC 311 complaints using topic modeling and extractive text summarization.

The system processes complaint descriptions, identifies common topics using Latent Dirichlet Allocation (LDA), and generates representative summaries using TF-IDF and cosine similarity.

## Objectives

1. Preprocess raw complaint descriptions.
2. Identify major complaint topics using LDA.
3. Generate representative complaint summaries using extractive summarization.
4. Evaluate the model using NLP evaluation metrics.

## Methodology

### 1. Text Preprocessing
- Handle missing complaint descriptions.
- Convert text to lowercase.
- Remove unnecessary characters and English stopwords.
- Prepare cleaned text for modeling.

### 2. Topic Modeling
- Apply CountVectorizer to convert text into numerical features.
- Train an LDA model with four topics.
- Identify important keywords and assign complaints to topics.

### 3. Extractive Text Summarization
- Apply TF-IDF vectorization.
- Calculate topic-level representative text using cosine similarity.
- Select representative complaint descriptions for each topic.

### 4. Model Evaluation
The system evaluates topic modeling and summarization using:
- LDA Perplexity
- Topic Diversity
- Average Topic Assignment Confidence
- Average Summary Relevance

## Dataset

**Dataset:** NYC 311 Complaints

- Original records: 369
- Processed records: 366
- Removed records: 3
- Number of topics: 4

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gensim
- LDA
- TF-IDF
- Cosine Similarity

## Evaluation Results

| Metric | Result |
|---|---:|
| LDA Perplexity | 293.686213 |
| Topic Diversity | 1.000000 |
| Average Topic Assignment Confidence | 0.611134 |
| Average Summary Relevance | 0.318100 |

## Project Files

- `01_preprocessing.py` – Text preprocessing
- `02_topic_modeling.py` – LDA topic modeling
- `03_text_summarization.py` – Extractive summarization
- `04_evaluation.py` – Model evaluation
- `requirements.txt` – Python dependencies
- `outputs/` – Generated models, summaries and evaluation results
- `CivicSift_AI_Week5_Report.pdf` – Detailed project report

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts in the following order:

```bash
python 01_preprocessing.py
python 02_topic_modeling.py
python 03_text_summarization.py
python 04_evaluation.py
```

Ensure the dataset is available in the expected `data/` directory before running the scripts.

## Limitations

- The dataset is relatively small.
- Short complaint descriptions can make topic identification difficult.
- Some topics may contain mixed or overlapping complaint categories.
- The summaries are representative extracts, not newly generated sentences.

## Conclusion

This project demonstrates how NLP techniques can be used to organize large collections of complaints into meaningful topics and generate representative summaries. The implementation provides a foundation for further improvements in automated complaint analysis.

