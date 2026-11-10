# CivicSift AI
### An Evidence-Aware Multilingual NLP Framework for Citizen Complaint Analysis

## Project Overview

CivicSift AI is an NLP-based citizen complaint analysis project.

The project explores how Natural Language Processing (NLP) and machine learning techniques can help analyse unstructured citizen complaints, identify recurring issues, classify complaint categories, and generate meaningful insights.

The project uses publicly available NYC 311 complaint data as its experimental dataset. It progressively explores text preprocessing, machine learning-based text classification, Named Entity Recognition (NER) pipeline design, topic modelling, and extractive text summarisation.

The long-term vision is to extend the framework towards multilingual and code-mixed complaint analysis, evidence-aware urgency estimation, duplicate detection, and human-reviewed complaint management.

**Important:** The current implementation is an experimental prototype. Multilingual processing, urgency estimation, duplicate detection, and a fully integrated dashboard are proposed future extensions, not completed features.

---

## Problem Statement

Citizen complaints are submitted in different formats, languages, and writing styles. Analysing large volumes of unstructured complaint text manually can be time-consuming and may make it difficult to identify recurring issues and patterns.

CivicSift AI explores how NLP techniques can support the organisation and analysis of complaint data through text classification, topic discovery, and automated summarisation.

---

## Project Objectives

- Explore NLP techniques for processing citizen complaint descriptions.
- Develop and evaluate machine learning-based text classification models.
- Design a Named Entity Recognition pipeline for extracting relevant information.
- Discover recurring complaint patterns using topic modelling.
- Generate representative complaint summaries using extractive summarisation.
- Evaluate implemented approaches and document their limitations.
- Explore future extensions for multilingual and human-reviewed complaint analysis.

---

## Dataset

**Source:** NYC Open Data – 311 Service Requests

Dataset endpoint:

[NYC 311 Service Requests API](https://data.cityofnewyork.us/resource/erm2-nwe9.csv)

The project uses complaint information from the NYC 311 public dataset.

Selected fields include:
- `complaint_type`
- `descriptor`

The dataset was used in separate experiments, with different preprocessing and filtering stages.

### Dataset usage across experiments

| Experiment | Dataset records |
|---|---:|
| Week 3 – Initial data fetched | 369 |
| Week 3 – Valid records after cleaning | 203 |
| Week 3 – Complaint categories | 24 |
| Week 5 – Initial records | 369 |
| Week 5 – Processed records | 366 |
| Week 5 – Removed records | 3 |

The record counts differ because the experiments used different cleaning and filtering requirements.

---

## Technologies and Tools

### Implemented

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- CountVectorizer
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Latent Dirichlet Allocation (LDA)
- Cosine Similarity
- Joblib
- Matplotlib

### Considered for Future Development

- IndicBERT
- Multilingual BERT (mBERT)
- XLM-RoBERTa
- Sentence Transformers
- Streamlit

These technologies are candidates for future development and are not claimed as completed integrations.

---

## Weekly Project Development

### Week 1 – NLP Project Planning and Proposal

**Objective:** Define the problem, proposed application, objectives, and development approach.

Work completed:
- Identified the problem of analysing unstructured citizen complaints.
- Defined the CivicSift AI project concept.
- Established project objectives and proposed features.
- Identified NLP techniques suitable for complaint analysis.
- Prepared the initial project proposal.

**Status:** Completed – Planning and documentation.

### Week 2 – Data Collection, Preprocessing and Feature Engineering

**Objective:** Develop the dataset strategy and document the preprocessing methodology.

Work completed:
- Identified NYC 311 public complaint data as the experimental data source.
- Documented relevant dataset fields and collection methodology.
- Prepared a text preprocessing workflow.
- Explored text cleaning, tokenisation, stop-word removal, and feature extraction approaches.
- Documented the proposed feature engineering methodology.

**Status:** Completed – Methodology and documentation.

**Note:** Week 2 focused on documenting the methodology. It should not be interpreted as a separate fully implemented end-to-end preprocessing system.

### Week 3 – Text Classification and Model Evaluation

**Objective:** Implement and compare machine learning models for complaint classification.

Implementation:
- Retrieved 369 complaint records.
- Removed records with missing required values, resulting in 203 usable records.
- Selected `descriptor` as the input text and `complaint_type` as the target.
- Identified 24 complaint categories.
- Applied a stratified 75:25 train-test split.
- Used TF-IDF with unigrams and bigrams.
- Implemented three classification algorithms.

**Experimental results:**

| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes | 37.25% |
| Logistic Regression | 43.14% |
| Linear SVM | 49.02% |

**Best-performing model:** Linear SVM.

Additional evaluation:
- Linear SVM weighted F1-score: 47.13%.
- Evaluation metrics included accuracy, weighted precision, weighted recall, and weighted F1-score.

**Key observation:** Linear SVM achieved the highest accuracy among the three tested models in this initial experiment.

**Limitations:**
- Small dataset.
- Imbalanced complaint categories.
- Short complaint descriptors rather than complete complaint narratives.
- English-language experimental data.
- No extensive hyperparameter tuning or cross-validation.

**Status:** Completed – Implementation and evaluation.

### Week 4 – Named Entity Recognition (NER) Pipeline Design

**Objective:** Design an NLP pipeline for identifying and extracting named entities from citizen complaints.

Work completed:
- Defined a proposed NER architecture.
- Identified general entity categories such as PERSON, LOCATION, ORGANIZATION, DATE, and TIME.
- Proposed civic-specific entities such as ISSUE and INFRASTRUCTURE.
- Explored BIO tagging and entity annotation approaches.
- Considered multilingual transformer models such as mBERT, XLM-RoBERTa, and IndicBERT.
- Documented entity validation, post-processing, confidence thresholds, and human review.

**Important limitation:** This week involved pipeline design and documentation. A trained or evaluated NER model was not implemented.

The NYC 311 dataset used in the experiments does not contain the token-level entity annotations needed to train and evaluate the proposed NER system directly.

**Status:** Completed – Pipeline design and documentation.

### Week 5 – Topic Modeling and Extractive Text Summarization

**Objective:** Discover recurring complaint themes and generate representative summaries.

Implementation:
- Processed 369 complaint records.
- Retained 366 processed records after cleaning.
- Applied text cleaning, lowercasing, and English stop-word removal.
- Used CountVectorizer for document-term matrix generation.
- Implemented Latent Dirichlet Allocation (LDA).
- Generated four topics.
- Applied TF-IDF and cosine similarity for extractive summarisation.
- Evaluated topic modelling and summary relevance.

**Topic modelling configuration:**
- Number of topics: 4
- Maximum vocabulary features: 1,000
- Minimum document frequency: 2
- Maximum document frequency: 0.95
- N-gram range: (1, 2)
- Document-term matrix shape: 366 × 195

**Topic distribution:**

| Topic | Complaints | Distribution |
|---|---:|---:|
| Topic 1 | 155 | 42.35% |
| Topic 2 | 66 | 18.03% |
| Topic 3 | 67 | 18.31% |
| Topic 4 | 78 | 21.31% |

**Evaluation results:**

| Metric | Result |
|---|---:|
| LDA Perplexity | 293.686213 |
| Topic Diversity | 1.000000 |
| Average Topic Assignment Confidence | 0.611134 |
| Average Summary Relevance | 0.318100 |

**Summary relevance by topic:**

| Topic | Relevance |
|---|---:|
| Topic 1 | 0.2660 |
| Topic 2 | 0.3278 |
| Topic 3 | 0.3468 |
| Topic 4 | 0.3318 |

The summarisation approach selected representative complaint descriptions using TF-IDF features, topic centroids, and cosine similarity.

**Limitations:**
- Some topics contain mixed or overlapping complaint themes.
- The dataset is relatively small and contains short descriptions.
- Topic diversity is a code-defined metric and does not establish semantic coherence.
- No ROUGE evaluation was performed because reference summaries were unavailable.
- Four topics were used as an initial experiment, not as a proven optimal configuration.

**Status:** Completed – Implementation and evaluation.

### Week 6 – Final Project Report and Documentation

**Objective:** Consolidate the work performed across all five previous weekly tasks.

Work completed:
- Reviewed and consolidated the five weekly reports.
- Documented the project methodology and experimental workflow.
- Presented the actual Week 3 classification results.
- Included the Week 4 NER design and its implementation limitations.
- Documented Week 5 topic modelling and summarisation results.
- Consolidated the project architecture, evaluation, limitations, and future scope.
- Prepared the final project report.

**Status:** Completed – Final documentation.

---

## Current Project Status

| Internship Task | Work Completed | Status |
|---|---|---|
| Week 1 | NLP project proposal | Completed |
| Week 2 | Data preprocessing methodology | Completed |
| Week 3 | Text classification and model evaluation | Completed |
| Week 4 | NER pipeline design | Completed |
| Week 5 | Topic modelling and extractive summarisation | Completed |
| Week 6 | Consolidated final project report | Completed |

---

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
├── Week-4-NER-Pipeline-Design/
│   ├── README.md
│   └── Week 4 Report
│
├── Week-5-Topic-Modeling-and-Summarization/
│   ├── README.md
│   ├── Python implementation files
│   ├── dataset and generated outputs
│   └── Week 5 Report
│
├── Week-6-Final-Report/
│   ├── README.md
│   ├── CivicSift_AI_Week6_Final_Report.docx
│   └── CivicSift_AI_Week6_Final_Report.pdf
│
└── README.md
```

*The structure above describes the intended organisation. File and folder names may vary slightly depending on the uploaded repository contents.*

---

## Key Findings

1. Linear SVM performed best among the three Week 3 classification models, achieving 49.02% accuracy.
2. TF-IDF provided a feature representation for the complaint classification experiment.
3. LDA identified four preliminary complaint themes in Week 5.
4. TF-IDF and cosine similarity were used to select representative complaint descriptions for extractive summaries.
5. The experiments demonstrate the potential of NLP techniques for analysing complaint data, while also highlighting the need for larger datasets and further validation.

---

## Limitations and Responsible Use

CivicSift AI is currently an academic experimental prototype, not a production-ready public complaint management platform.

Key limitations include:
- Limited dataset size and representativeness.
- Dependence on English-language NYC 311 data for the implemented experiments.
- No demonstrated multilingual classification performance.
- No trained NER model or token-level entity evaluation.
- No implemented urgency estimation or duplicate detection.
- No fully integrated dashboard or deployed end-to-end application.
- Topic modelling and summarisation results require further validation.

Any future deployment should incorporate privacy protection, secure data handling, human review, and transparent model evaluation. Automated predictions should support, rather than replace, human decision-making.

---

## Future Scope

Potential future improvements include:

- Extend complaint classification to Hindi, Kannada, English, and Romanised or code-mixed text.
- Evaluate multilingual transformer-based models.
- Implement and evaluate a trained NER system using appropriately annotated data.
- Develop evidence-aware complaint urgency estimation.
- Explore semantic similarity and duplicate complaint detection.
- Improve topic coherence using larger and more representative datasets.
- Develop an interactive dashboard using Streamlit.
- Integrate human-in-the-loop review and feedback.
- Explore secure and responsible deployment.

---

## Conclusion

CivicSift AI explores the application of Natural Language Processing and machine learning to citizen complaint analysis through a series of planned and implemented internship tasks.

The project progressed from initial planning and preprocessing methodology to text classification, NER pipeline design, topic modelling, extractive summarisation, and consolidated documentation.

The experimental results provide an initial foundation for further development. Although several proposed features remain future work, the completed tasks demonstrate practical exploration of multiple NLP techniques and their evaluation on public complaint data.

---

## References

1. NYC Open Data – [311 Service Requests](https://data.cityofnewyork.us/resource/erm2-nwe9.csv)
2. Scikit-learn Documentation – [https://scikit-learn.org/](https://scikit-learn.org/)
3. Pandas Documentation – [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
4. Python Documentation – [https://docs.python.org/3/](https://docs.python.org/3/)

---

**Developed by:** Riya Desai  
**Academic Domain:** Artificial Intelligence and Machine Learning  
**Project Domain:** Natural Language Processing  
**Project Status:** Six-Week Internship Tasks Completed – Further Development Proposed
