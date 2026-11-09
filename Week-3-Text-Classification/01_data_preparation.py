import pandas as pd

# NYC Open Data API
url = (
    "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
    "?$select=complaint_type,descriptor"
    "&$where=complaint_type%20IS%20NOT%20NULL"
    "&$limit=5000"
)

print("Downloading public complaint dataset...")

try:
    df = pd.read_csv(url)

    # Combine the available text description fields
    df["text"] = df["descriptor"].fillna("").astype(str)

    # Keep only the text and target category
    df = df[["text", "complaint_type"]]

    # Clean the dataset
    df["text"] = df["text"].str.strip()
    df["complaint_type"] = df["complaint_type"].str.strip()

    df = df.dropna()
    df = df[df["text"] != ""]
    df = df.drop_duplicates()

    # Save the dataset
    df.to_csv("data/nyc_311_complaints.csv", index=False)

    print("\nDataset downloaded successfully!")
    print("Total records:", len(df))
    print("\nComplaint categories:")
    print(df["complaint_type"].value_counts().head(10))

    print("\nSample records:")
    print(df.head())

except Exception as e:
    print("Error:", e)
