## 📝 Smart Essay Scorer

### A machine learning web app built with Python, NLP, and Streamlit that automatically scores essays based on writing quality. This project demonstrates text preprocessing, feature engineering, ML model training, and deployment using Streamlit.

---
## 🚀 Features

 - 📂 Multiple Input Options: Paste essay text or upload a .txt file.

 - 🧹 Automated Preprocessing: Cleans text (stopword removal, lemmatization, etc.).

 - 📊 Essay Metrics: Displays word count, sentence count, avg. word length, and more.

 - 🤖 ML Model Predictions: Provides essay scores using trained machine learning models.

 - 🔍 Explainability Ready: Designed for future integration of explainable AI (e.g., SHAP).

 - 🌐 Interactive UI: Built with Streamlit for smooth user experience.

---
##  Tech Stack

Python 3.10+

Libraries:

 - streamlit – Web app UI

 - nltk – NLP preprocessing (tokenization, lemmatization, stopwords)

 - scikit-learn – Machine Learning models & feature extraction

 - joblib – Model persistence

 - numpy, scipy – Vectorization and numeric processing

---
## 📂 Project Structure

```
Smart_Essay_Scorer/
│── app.py                # Main Streamlit app
│── model.pkl             # Trained ML model (joblib)
│── vectorizer.pkl        # Feature extraction pipeline
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── sample_essays/        # Example essay text files

```

## Installation & Usage

1. Clone the repository

```
git clone https://github.com/Md-Faiz-Alam/Smart_Essay_Scorer.git
cd Smart_Essay_Scorer

```

2. Create virtual environment (recommended)

```
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run the Streamlit app

```
streamlit run app.py
```

---
## 📊 Example Output

1. Input Essay → Paste or upload

2. Metrics → Word count: 514, Avg sentence length: 18.3, Avg word length: 7.5

3. Predicted Score → 5.0 (Rounded)

---

## 🎯 Learning Outcomes

1. This project helped me practice:

2. NLP preprocessing (tokenization, lemmatization, stopwords).

3. Feature engineering using Bag-of-Words/TF-IDF.

4. ML model training (Regression/Classification for scoring).

5. Deployment skills with Streamlit.

6. Version control & project structuring for real-world applications.

---

## 📌 Future Improvements

 - Add deep learning models (e.g., LSTMs, Transformers).

 - Integrate SHAP/LIME explainability.

 - Deploy on Streamlit Cloud/Heroku.

 - Enhance UI with better essay visualization.

---

## 👤 Author

### Md Faiz Alam
📧 Email: [mail me](mailto:mdfaiz3388@gmail.com)
💼 LinkedIn: [Md_Faiz_Alam](https://www.linkedin.com/in/alammdfaiz)
