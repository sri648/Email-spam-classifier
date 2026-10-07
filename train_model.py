import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"],
    encoding="latin-1"
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("\nFirst 5 rows:")
print(data.head())


# 2. Separate input and output
X = data["message"]
y = data["label"]


# 3. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5. Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# 6. Make predictions
y_pred = model.predict(X_test_tfidf)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Save the trained model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved as spam_model.pkl")
print("Vectorizer saved as vectorizer.pkl")