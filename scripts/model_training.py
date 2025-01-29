from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os 

from data_preprocessing import preprocessing
import pandas as pd
if not os.path.exists('model'):
    os.makedirs('model')
def train_model(x_train,y_train):
    vectorizer=TfidfVectorizer()
    x_train_vec=vectorizer.fit_transform(x_train)
    model=RandomForestClassifier()
    model.fit(x_train_vec,y_train)
    return  vectorizer,model


if __name__=="__main__":
    data=pd.read_csv('data/train_sarcasm.csv',encoding='latin-1')
    x_train,x_test,y_train,y_test=preprocessing(data)
    vectorizer,model=train_model(x_train,y_train)
    with open("model/vectorizer.pkl",'wb') as vec_file:
        pickle.dump(vectorizer,vec_file)
    with open("model/model_train.pkl",'wb') as model_file:
        pickle.dump(model,model_file)
    print('all process is ended')

    


