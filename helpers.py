import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import joblib

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def predict_label(review_title, review_text):
    # Load the Naive Bayes model and its vectorizer
    naive_bayes_classifier = joblib.load('Naive_Bayes.pkl')
    tfidf_vectorizer = joblib.load('Naive_Bayes_Vectorizer.pkl')
    
    # Preprocess the input data
    combined_text = preprocess_text(review_title + ' ' + review_text)
    
    # Transform the preprocessed text using the vectorizer
    X_test = tfidf_vectorizer.transform([combined_text])
    
    # Make predictions using the loaded model
    predicted_label = naive_bayes_classifier.predict(X_test)
    
    return predicted_label[0]
