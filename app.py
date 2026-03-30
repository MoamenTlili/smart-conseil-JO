from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Load the models
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("svm_model.pkl")

# API initialization
app = FastAPI(title="Smart Conseil Harassment Detection API")

# input schema
class TextRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)        
    return text.strip()

@app.post("/predict")
def predict_harassment(request: TextRequest):
    # Preprocess
    cleaned = clean_text(request.text)
    
    # Vectorize
    vec_text = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vec_text)[0]
    
    return {
        "original_text": request.text,
        "is_harassment": bool(prediction == 1),
        "status": "success"
    }

@app.get("/health")
def health_check():
    return {"status": "API and Models are loaded successfully."}