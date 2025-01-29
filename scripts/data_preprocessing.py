import pandas as pd
from sklearn.model_selection import train_test_split
def preprocessing(df):
    df['Text']=df['Text'].str.lower()
    return train_test_split(df['Text'],df['Label'],test_size=0.2,random_state=42)
 

if __name__=="__main__":
    df=pd.read_csv('data/train_sarcasm.csv',encoding='latin-1')
    x_train,x_test,y_train,y_test=preprocessing(df)
    print(f'Data Splited Successfully {len(x_train)},{len(x_test)}')