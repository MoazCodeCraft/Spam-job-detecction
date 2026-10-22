
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_jobs.csv")

# Separate input text and target labels
X = df["job_text"]
y = df["fraudulent"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert text into numerical features
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 1),
    sublinear_tf=True,
    dtype="float32"
)

# Learn vocabulary from training data only
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Create the model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
print("\nTraining the model...")
model.fit(X_train_tfidf, y_train)

# Create folder for saved model
os.makedirs("saved_model", exist_ok=True)

# Save both the trained model and TF-IDF vectorizer
joblib.dump(
    {
        "model": model,
        "tfidf": tfidf
    },
    "saved_model/spam_job_model.pkl"
)

print("\nModel saved successfully!")

print("Model training completed!")

# Predict on unseen test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Display results
print("Total job postings:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining feature matrix:", X_train_tfidf.shape)
print("Testing feature matrix:", X_test_tfidf.shape)

print("\nNumber of vocabulary features:", len(tfidf.vocabulary_))

print("\nTF-IDF conversion completed successfully!")
