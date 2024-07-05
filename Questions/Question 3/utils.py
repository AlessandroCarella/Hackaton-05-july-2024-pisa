import pandas as pd
from sklearn.model_selection import train_test_split


def getDataset():
    df=pd.read_csv('../../Dataset/final dataset.csv')

    train_cols = df.columns.difference(['customer_types', 'conversion', 'offer', 'treat'])
    X = df[train_cols] 
    y = df['treat']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=12)
    return X_train, X_test, Y_train, Y_test

def get_names():
    X_train, X_test, Y_train, Y_test = getDataset()
    train_names = X_train.index.tolist()
    test_names = X_test.index.tolist()
    return train_names, test_names

