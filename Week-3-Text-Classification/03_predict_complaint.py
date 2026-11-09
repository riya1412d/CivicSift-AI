
import joblib

# Load the trained Linear SVM model
model = joblib.load("models/linear_svm.joblib")

print("=" * 50)
print("       CIVICSIFT AI - COMPLAINT CLASSIFIER")
print("=" * 50)

while True:
    complaint = input("\nEnter a complaint (or type 'exit'): ")

    if complaint.lower() == "exit":
        print("Thank you for using CivicSift AI!")
        break

    if not complaint.strip():
        print("Please enter a valid complaint.")
        continue

    # Predict complaint category
    prediction = model.predict([complaint])[0]

    print("\nPredicted Complaint Category:", prediction)
    print("-" * 50)
