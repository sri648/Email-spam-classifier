import streamlit as st
import joblib


# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Page title
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧"
)

st.title("📧 Email Spam Classifier")
st.write("Enter a message below to check whether it is Spam or Ham.")


# User input
message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste your email/message here..."
)


# Prediction button
if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Convert message into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_tfidf)[0]

        # Display result
        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
            st.write("This message is classified as spam.")

        else:
            st.success("✅ HAM — NOT SPAM")
            st.write("This message is classified as a normal message.")