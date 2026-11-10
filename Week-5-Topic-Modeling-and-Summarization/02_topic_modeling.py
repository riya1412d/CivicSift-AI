
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "preprocessed_complaints.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

print("=" * 60)
print("CIVICSIFT AI - LDA TOPIC MODELING")
print("=" * 60)

# 1. Load preprocessed dataset
df = pd.read_csv(DATA_PATH)

documents = df["clean_text"].fillna("").astype(str)

print("\nDataset loaded successfully!")
print("Total documents:", len(documents))

# 2. Convert text into numerical features
vectorizer = CountVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)
)

document_term_matrix = vectorizer.fit_transform(documents)

print("Document-term matrix shape:", document_term_matrix.shape)
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# 3. Train LDA model
number_of_topics = 4

lda_model = LatentDirichletAllocation(
    n_components=number_of_topics,
    random_state=42,
    max_iter=20,
    learning_method="batch"
)

lda_model.fit(document_term_matrix)

print("\nLDA model trained successfully!")

# 4. Display topics
feature_names = vectorizer.get_feature_names_out()
topic_results = []

print("\nDISCOVERED TOPICS")
print("-" * 60)

for topic_index, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_indices]

    topic_results.append({
        "topic": topic_index + 1,
        "keywords": ", ".join(top_words)
    })

    print(f"\nTopic {topic_index + 1}:")
    print(", ".join(top_words))

# 5. Assign dominant topic to each complaint
topic_distributions = lda_model.transform(document_term_matrix)

df["dominant_topic"] = topic_distributions.argmax(axis=1) + 1
df["topic_probability"] = topic_distributions.max(axis=1)

# 6. Save topic results
topic_df = pd.DataFrame(topic_results)
topic_df.to_csv(OUTPUT_DIR / "lda_topics.csv", index=False)

df.to_csv(OUTPUT_DIR / "complaints_with_topics.csv", index=False)

# 7. Save trained model and vectorizer
joblib.dump(lda_model, OUTPUT_DIR / "lda_model.pkl")
joblib.dump(vectorizer, OUTPUT_DIR / "count_vectorizer.pkl")

# 8. Display topic distribution
print("\nTOPIC DISTRIBUTION")
print("-" * 60)
print(df["dominant_topic"].value_counts().sort_index())

print("\nSample complaint-topic assignments:")
print(
    df[["text", "dominant_topic", "topic_probability"]]
    .head(10)
    .to_string(index=False)
)

print("\nAll topic modeling outputs saved successfully!")
