import pickle

def predict_sentiment(text):
    with open("model/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("model/model_train.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

if __name__ == "__main__":
    result = predict_sentiment("This is amazing!")
    print(f"Sentiment: {result}")
