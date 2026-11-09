
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load the dataset
df = pd.read_csv("data/nyc_311_complaints.csv")

# 2. Keep categories with at least 5 examples
counts = df["complaint_type"].value_counts()
df = df[df["complaint_type"].isin(counts[counts >= 5].index)]

print("Records after filtering:", len(df))
print("Number of categories:", df["complaint_type"].nunique())

X = df["text"]
y = df["complaint_type"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Define three classification models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

results = []

# 5. Train and evaluate each model
for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n", "=" * 45)
    print("MODEL:", name)
    print("=" * 45)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, "models/" + filename)

# 6. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 7. Plot model comparison
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("NLP Classification Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.show()

print("\nAll models trained and saved successfully!")
