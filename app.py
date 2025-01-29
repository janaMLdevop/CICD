from fastapi import FastAPI
from scripts.inference import predict_sentiment
app=FastAPI()
@app.get('/predict')
def predict(text:str):
    sentiment=predict_sentiment(text)
    return {"text":text,"sentiment":sentiment}

