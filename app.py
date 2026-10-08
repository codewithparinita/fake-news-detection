import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to predict whether it is Fake or Real."
)

news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)

if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:
        cleaned_news = clean_text(news)
        news_tfidf = tfidf.transform([cleaned_news])

        prediction = model.predict(news_tfidf)[0]
        probabilities = model.predict_proba(news_tfidf)[0]

        confidence = max(probabilities) * 100

        if prediction == 0:
            st.error("❌ FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.info(f"Confidence: {confidence:.2f}%")
