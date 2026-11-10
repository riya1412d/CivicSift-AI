
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = OUTPUT_DIR / "complaints_with_topics.csv"
TOPICS_PATH = OUTPUT_DIR / "lda_topics.csv"
SUMMARY_PATH = OUTPUT_DIR / "topic_summaries.csv"

MODEL_PATH = OUTPUT_DIR / "lda_model.pkl"
VECTORIZER_PATH = OUTPUT_DIR / "count_vectorizer.pkl"

print("=" * 60)
print("CIVICSIFT AI - MODEL EVALUATION")
print("=" * 60)

# 1. Load saved data and models
df = pd.read_csv(DATA_PATH)
topics_df = pd.read_csv(TOPICS_PATH)
summaries_df = pd.read_csv(SUMMARY_PATH)

lda_model = joblib.load(MODEL_PATH)
count_vectorizer = joblib.load(VECTORIZER_PATH)

documents = df["clean_text"].fillna("").astype(str)

X = count_vectorizer.transform(documents)

print("\nDataset and trained models loaded successfully!")

# 2. Evaluate LDA perplexity
perplexity = lda_model.perplexity(X)

# 3. Calculate topic diversity
all_topic_words = []

for keywords in topics_df["keywords"]:
    words = [word.strip() for word in keywords.split(",")]
    all_topic_words.extend(words)

unique_words = set(all_topic_words)

topic_diversity = len(unique_words) / len(all_topic_words)

# 4. Calculate topic assignment confidence
average_confidence = df["topic_probability"].mean()

# 5. Calculate summary relevance
summary_scores = []

for _, summary in summaries_df.iterrows():

    topic_number = summary["topic"]

    topic_data = df[
        df["dominant_topic"] == topic_number
    ].drop_duplicates(subset=["text"])

    complaints = (
        topic_data["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    selected_complaints = [
        item.strip()
        for item in str(
            summary["representative_complaints"]
        ).split(";")
        if item.strip()
    ]

    if len(complaints) == 0 or len(selected_complaints) == 0:
        continue

    tfidf = TfidfVectorizer()

    topic_matrix = tfidf.fit_transform(complaints)

    centroid = np.asarray(topic_matrix.mean(axis=0))

    selected_matrix = tfidf.transform(selected_complaints)

    relevance = cosine_similarity(
        selected_matrix,
        centroid
    ).flatten()

    average_relevance = relevance.mean()

    summary_scores.append({
        "topic": topic_number,
        "summary_relevance": average_relevance
    })

# 6. Prepare evaluation results
evaluation_results = {
    "Metric": [
        "LDA Perplexity",
        "Topic Diversity",
        "Average Topic Assignment Confidence",
        "Average Summary Relevance"
    ],
    "Score": [
        perplexity,
        topic_diversity,
        average_confidence,
        np.mean([
            item["summary_relevance"]
            for item in summary_scores
        ]) if summary_scores else 0
    ]
}

evaluation_df = pd.DataFrame(evaluation_results)

# 7. Save evaluation results
evaluation_df.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

pd.DataFrame(summary_scores).to_csv(
    OUTPUT_DIR / "summary_relevance_by_topic.csv",
    index=False
)

# 8. Display results
print("\nMODEL EVALUATION RESULTS")
print("-" * 60)
print(evaluation_df.to_string(index=False))

print("\nSUMMARY RELEVANCE BY TOPIC")
print("-" * 60)

for item in summary_scores:
    print(
        f"Topic {item['topic']}: "
        f"{item['summary_relevance']:.4f}"
    )

print("\nEvaluation files saved successfully!")
print("Output folder:", OUTPUT_DIR)

print("\nEvaluation completed successfully!")
