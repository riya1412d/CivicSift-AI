
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "complaints_with_topics.csv"
TOPICS_PATH = BASE_DIR / "outputs" / "lda_topics.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

print("=" * 60)
print("CIVICSIFT AI - EXTRACTIVE TEXT SUMMARIZATION")
print("=" * 60)

# 1. Load topic-assigned complaints
df = pd.read_csv(DATA_PATH)
topics_df = pd.read_csv(TOPICS_PATH)

print("\nDataset loaded successfully!")
print("Total complaints:", len(df))

# 2. Prepare summary storage
summaries = []

# 3. Generate summaries for each discovered topic
for topic_number in sorted(df["dominant_topic"].unique()):

    topic_data = df[
        df["dominant_topic"] == topic_number
    ].copy()

    # Remove duplicate complaint descriptions
    topic_data = topic_data.drop_duplicates(
        subset=["text"]
    )

    complaints = (
        topic_data["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    if len(complaints) == 0:
        continue

    print(f"\nProcessing Topic {topic_number}...")

    # 4. Convert complaints into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(complaints)

    # 5. Calculate average TF-IDF representation
    # Convert NumPy matrix into a standard array
    topic_centroid = np.asarray(
        tfidf_matrix.mean(axis=0)
    )

    # 6. Calculate similarity scores
    similarity_scores = cosine_similarity(
        tfidf_matrix,
        topic_centroid
    ).flatten()

    topic_data["summary_score"] = similarity_scores

    # 7. Select five most representative complaints
    representative_complaints = (
        topic_data.sort_values(
            by="summary_score",
            ascending=False
        )
        .head(5)
    )

    # 8. Retrieve topic keywords
    topic_keywords = topics_df.loc[
        topics_df["topic"] == topic_number,
        "keywords"
    ].iloc[0]

    summary_text = "; ".join(
        representative_complaints["text"].tolist()
    )

    # 9. Find frequent complaint categories
    if "complaint_type" in topic_data.columns:

        top_categories = (
            topic_data["complaint_type"]
            .value_counts()
            .head(3)
            .index.tolist()
        )

        categories = ", ".join(top_categories)

    else:
        categories = "Not available"

    # 10. Store summary
    summaries.append({
        "topic": topic_number,
        "topic_keywords": topic_keywords,
        "representative_complaints": summary_text,
        "top_complaint_categories": categories,
        "complaints_in_topic": len(topic_data)
    })

    # 11. Display summary
    print("\nTOPIC", topic_number, "SUMMARY")
    print("-" * 50)

    print("Keywords:", topic_keywords)

    print("Representative complaints:")

    for complaint in representative_complaints["text"]:
        print("-", complaint)

    print("Top complaint categories:", categories)

# 12. Save summaries as CSV
summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    OUTPUT_DIR / "topic_summaries.csv",
    index=False
)

# 13. Save summaries as readable text file
with open(
    OUTPUT_DIR / "topic_summaries.txt",
    "w",
    encoding="utf-8"
) as file:

    for summary in summaries:

        file.write(f"\nTOPIC {summary['topic']}\n")
        file.write("-" * 50 + "\n")

        file.write(
            f"Keywords: {summary['topic_keywords']}\n"
        )

        file.write(
            f"Representative complaints: "
            f"{summary['representative_complaints']}\n"
        )

        file.write(
            f"Top complaint categories: "
            f"{summary['top_complaint_categories']}\n"
        )

        file.write(
            f"Number of complaints: "
            f"{summary['complaints_in_topic']}\n"
        )

# 14. Final results
print("\n" + "=" * 60)
print("TEXT SUMMARIZATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nTotal topics summarized:", len(summary_df))

print("Generated files:")
print("- topic_summaries.csv")
print("- topic_summaries.txt")

print("\nOutput location:", OUTPUT_DIR)
