from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import helpers

app = FastAPI()

# Load Naive Bayes model and its vectorizer
nb_classifier = joblib.load('Naive_Bayes.pkl')
tfidf_vectorizer = joblib.load('Naive_Bayes_Vectorizer.pkl')

# Define input model
class MovieInput(BaseModel):
    title: str
    overview: str

@app.post("/move-genreify")
def predict(movie_input: MovieInput):
    title = movie_input.title
    overview = movie_input.overview

    if not title or not overview:
        raise HTTPException(status_code=400, detail="Title and overview are required.")

    processed_input = helpers.preprocess_text(title + ' ' + overview)
    input_vectorized = tfidf_vectorizer.transform([processed_input])
    predicted_label = nb_classifier.predict(input_vectorized)

    return {"predicted_label": predicted_label[0]}

@app.options("/move-genreify")
async def get_move_genreify_options():
    return {"message": "Allowed Methods: POST"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
