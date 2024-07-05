import pandas as pd
from sklearn.model_selection import train_test_split

def get_X_train ():
    return pd.read_csv('../Dataset/X_train.csv')

def get_X_test ():
    return pd.read_csv('../Dataset/X_test.csv')

def get_Y_train ():
    return pd.read_csv('../Dataset/Y_train.csv')

def get_Y_test ():
    return pd.read_csv('../Dataset/Y_test.csv')


def get_All ():
    
    X_train = get_X_train()
    X_test = get_X_test()
    Y_train = get_Y_train() 
    Y_test = get_Y_test()

    return X_train, X_test, Y_train, Y_test


def get_Datasets_With_25_Split():
    df=pd.read_csv('../Dataset/datasetWithTarget.csv')

    train_cols = df.columns.difference(['customer_types', 'conversion'])
    y = df[train_cols] 
    X = df['customer_types']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=12)
    return X_train, X_test, Y_train, Y_test