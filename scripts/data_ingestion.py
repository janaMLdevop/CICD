import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path,encoding='latin-1')

if __name__ == "__main__":
    df = load_data("data/train_sarcasm.csv")
    print(f"Loaded data with {len(df)} rows")
