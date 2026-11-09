# CivicSift AI

### An Evidence-Aware Multilingual NLP Framework for Citizen Complaint Analysis

## Project Overview

CivicSift AI is an NLP-based project focused on analysing citizen complaints and supporting efficient complaint management.

The proposed framework aims to process complaints written in multiple languages, including English, Hindi, Kannada, and Romanised or code-mixed text.

Its long-term goal is to classify complaints, estimate urgency, identify similar or duplicate complaints, and provide useful insights through an accessible dashboard.

## Problem Statement

Citizen complaints are often submitted in different languages, formats, and writing styles. Manually analysing and categorising these complaints can be time-consuming and may result in delayed responses or duplicate records.

CivicSift AI explores how Natural Language Processing and machine learning can help streamline this process.

## Objectives

- Classify citizen complaints into relevant categories.
- Estimate complaint urgency using textual evidence.
- Identify similar and potentially duplicate complaints.
- Explore multilingual and code-mixed text processing.
- Extract important information from complaint descriptions.
- Support human-reviewed complaint management.

## Key Features (Proposed)

- Multilingual complaint classification.
- Evidence-aware urgency estimation.
- Semantic similarity and duplicate detection.
- Named Entity Recognition (NER).
- Complaint analytics and visualisation dashboard.
- Human-in-the-loop review.

## Technologies

**Implemented so far:**
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Joblib
- Matplotlib

**Technologies considered for future development:**
- IndicBERT / XLM-R
- Sentence Transformers
- Streamlit

## Project Methodology

1. Project planning and NLP proposal development.
2. Dataset strategy, preprocessing methodology and feature engineering.
3. Text classification using machine learning algorithms.
4. Named Entity Recognition (NER).
5. Topic modelling and text summarisation.
6. Integration, evaluation and documentation.

## Current Progress

| Internship Task | Description | Status |
|---|---|---|
| Week 1 | NLP Project Planning and Proposal | Completed |
| Week 2 | Data Collection, Preprocessing and Feature Engineering Methodology | Completed |
| Week 3 | Text Classification and Model Evaluation | Completed |
| Week 4 | Named Entity Recognition | Upcoming |
| Week 5 | Topic Modelling and Summarisation | Upcoming |
| Week 6 | Project Integration and Final Documentation | Upcoming |

## Week 3 – Initial Experimental Results

Three machine learning models were implemented and evaluated using a sample of NYC 311 complaint data.

| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes | 37.25% |
| Logistic Regression | 43.14% |
| Linear SVM | 49.02% |

Linear SVM achieved the best accuracy in this initial experiment.

**Note:** These are preliminary results from a small English-language dataset. They do not establish multilingual performance or production readiness.

## Repository Structure

```text
CivicSift-AI/
│
├── Week-1-Project-Proposal/
│   ├── README.md
│   └── Week 1 Report
│
├── Week-2-Data-Preprocessing/
│   ├── README.md
│   └── Week 2 Report
│
├── Week-3-Text-Classification/
│   ├── README.md
│   ├── data/
│   ├── models/
│   ├── results/
│   └── Python implementation files
│
└── README.md
```

## Documentation

- [Week 1 – NLP Project Proposal](Week-1-Project-Proposal/)
- [Week 2 – Data Preprocessing and Feature Engineering](Week-2-Data-Preprocessing/)
- [Week 3 – Text Classification and Evaluation](Week-3-Text-Classification/)

## Future Scope

- Extend classification to multilingual and code-mixed complaints.
- Improve model performance using larger and more representative datasets.
- Implement Named Entity Recognition.
- Explore topic modelling and automated summarisation.
- Develop urgency estimation and duplicate detection modules.
- Build an integrated complaint analysis dashboard.

## Project Status

**In Progress – NLP Internship Project**

CivicSift AI is being developed incrementally through practical NLP tasks, experimentation, model evaluation, and documentation.

---

**Developed by:** Riya Desai  
**Domain:** Artificial Intelligence and Machine Learning | Natural Language Processing
