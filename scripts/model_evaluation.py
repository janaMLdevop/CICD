from sklearn.metrics import classification_report
import pickle

def evaluate_model(X_test, y_test):
    with open("model/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("model/model_train.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    from data_preprocessing import preprocessing
    import pandas as pd

    df = pd.read_csv("data/train_sarcasm.csv",encoding='latin-1')
    _, X_test, _, y_test = preprocessing(df)
    evaluate_model(X_test, y_test)
