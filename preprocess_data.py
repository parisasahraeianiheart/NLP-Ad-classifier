import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess advertisement dataset.
    """
    data = pd.read_csv(file_path)
    # Example preprocessing: lowercasing and splitting
    data['AdText'] = data['AdText'].str.lower()
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

if __name__ == "__main__":
    train, test = load_and_preprocess_data("data/ad_dataset.csv")
    print("Training data:\n", train.head())
    print("Test data:\n", test.head())
