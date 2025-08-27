import streamlit as st
import numpy as np
import joblib
import os
from scipy.sparse import hstack
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ------------------------------
# NLTK setup
# ------------------------------
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# ------------------------------
# Load models and vectorizer
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_forest = joblib.load(os.path.join(BASE_DIR, "models", "Random_forest_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
model_xg = joblib.load(os.path.join(BASE_DIR, "models", "model_xg.pkl"))
model_lgb = joblib.load(os.path.join(BASE_DIR, "models", "model_lgb.pkl"))

# ------------------------------
# Essay cleaning function
# ------------------------------
def clean_essay(text, remove_stopwords=True, lemmatize=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    if remove_stopwords:
        text = ' '.join([w for w in text.split() if w not in ENGLISH_STOP_WORDS])
    if lemmatize:
        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📝 Smart Essay Scorer")
st.write("Paste your essay, or upload a `.txt` file, then select a model to predict the essay score.")

# Initialize session state
if "essay_text" not in st.session_state:
    st.session_state.essay_text = ""

# ------------------------------
# Clear button callback
# ------------------------------
def clear_text():
    st.session_state.update({"essay_text": ""})

# Input options
input_option = st.radio("Choose input method:", ["Paste Essay", "Upload File"])

if input_option == "Paste Essay":
    st.text_area(
        "Paste your essay here:",
        key="essay_text",
        height=300
    )
else:
    uploaded_file = st.file_uploader("Upload your essay (.txt)", type=["txt"])
    if uploaded_file:
        st.session_state.essay_text = uploaded_file.read().decode("utf-8")
        st.text_area(
            "Uploaded essay content:",
            key="essay_text",
            height=300
        )

# ✅ Clear button (safe way)
st.button("Clear", on_click=clear_text)


# Now use essay_text
essay_text = st.session_state.essay_text

# Model selection
model_choice = st.selectbox("Select Model:", ["Random Forest", "XGBoost", "LightGBM"])

# Predict button
if st.button("Predict Score"):
    if not essay_text.strip():
        st.warning("Please provide an essay first.")
    else:
        # Clean essay
        essay_clean = clean_essay(essay_text)

        # Numeric features
        word_count = len(essay_text.split())
        sent_count = essay_text.count('.') + 1
        words_per_sentence = word_count / max(sent_count, 1)
        char_count = len(essay_text)
        avg_word_length = char_count / max(word_count, 1)
        X_numeric_new = np.array([[word_count, sent_count, word_count, words_per_sentence, char_count, avg_word_length]])

        # Text features
        X_text_new = tfidf.transform([essay_clean])

        # Combine numeric + text features
        X_new_final = hstack([X_numeric_new, X_text_new])

        # Select model
        if model_choice == "Random Forest":
            model = model_forest
        elif model_choice == "XGBoost":
            model = model_xg
        else:
            model = model_lgb

        # Predict
        predicted_score = model.predict(X_new_final)[0]
        predicted_score_rounded = round(predicted_score)

        # Display metrics
        st.subheader("📊 Essay Metrics")
        st.write(f"**Total Words:** {word_count}")
        st.write(f"**Total Sentences:** {sent_count}")
        st.write(f"**Average Words per Sentence:** {words_per_sentence:.2f}")
        st.write(f"**Average Word Length:** {avg_word_length:.2f}")

        st.subheader("📝 Predicted Essay Score")
        st.success(f"Predicted Score: {predicted_score:.2f}")
        st.info(f"Rounded Predicted Score: {predicted_score_rounded}")
