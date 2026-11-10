
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "nyc_311_complaints.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "preprocessed_complaints.csv"

print("=" * 60)
print("CIVICSIFT AI - WEEK 5 TEXT PREPROCESSING")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nDataset columns:", list(df.columns))

# 2. Check required column
if "text" not in df.columns:
    raise ValueError("The dataset must contain a 'text' column.")

# 3. Handle missing values
df["text"] = df["text"].fillna("").astype(str)

# 4. Define stop words
stop_words = set(ENGLISH_STOP_WORDS)

# 5. Text cleaning function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

# 6. Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# 7. Remove empty cleaned records
df = df[df["clean_text"].str.strip() != ""].copy()

# 8. Save processed dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# 9. Display results
print("\nFirst 10 processed complaints:")
print(df[["text", "clean_text"]].head(10).to_string(index=False))

print("\nPREPROCESSING SUMMARY")
print("-" * 40)
print("Original records:", len(pd.read_csv(DATA_PATH)))
print("Processed records:", len(df))
print("Empty cleaned records:", (df["clean_text"] == "").sum())
print("Output saved to:", OUTPUT_PATH)

print("\nText preprocessing completed successfully!")
