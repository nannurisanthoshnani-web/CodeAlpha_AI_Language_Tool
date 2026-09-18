import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQ Dataset
faqs = {
    "What is CodeAlpha?": "CodeAlpha is a software development company offering internships in emerging technologies.",
    "How many tasks do I need to submit?": "You must complete and submit at least 2 or 3 tasks to earn your certificate.",
    "Where do I submit my source code?": "You must upload your code to GitHub and share the link in the official submission form.",
    "Will I get a certificate?": "Yes, a QR-verified completion certificate will be issued upon successfully completing the tasks."
}

questions = list(faqs.keys())

st.title("FAQ Chatbot")

user_query = st.text_input("Ask a question about the internship:")

if user_query:
    # Vectorize and compute similarity
    vectorizer = TfidfVectorizer().fit(questions + [user_query])
    vectors = vectorizer.transform(questions + [user_query])
    
    # Calculate similarity scores
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_idx = similarity.argmax()
    score = similarity[0][best_match_idx]
    
    if score > 0.2:
        st.write("**Bot:**", faqs[questions[best_match_idx]])
    else:
        st.write("**Bot:** I'm sorry, I don't have an answer for that question. Please check the internship document.")