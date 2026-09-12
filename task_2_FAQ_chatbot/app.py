import json
import os
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load FAQ data
file_path = os.path.join(os.path.dirname(__file__), "faq_data.json")
with open(file_path, "r", encoding="utf-8") as file:
    faq_data = json.load(file)


questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]


# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)


def get_answer(user_question):

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.20:
        return "Sorry, I could not find an answer to your question."

    return answers[best_match]


# Streamlit interface
st.title("🤖 AI FAQ Chatbot")
st.write("Ask a question about the college.")

question = st.text_input("Enter your question:")

if question:
    answer = get_answer(question)

    st.success(answer)