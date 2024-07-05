import pandas as pd
from sklearn.model_selection import train_test_split


def get_Datasets_With_25_Split():
    df=pd.read_csv('../Dataset/final dataset.csv')

    train_cols = df.columns.difference(['customer_types', 'conversion', 'offer', 'treat'])
    X = df[train_cols] 
    y = df['treat']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=12)
    return X_train, X_test, Y_train, Y_test