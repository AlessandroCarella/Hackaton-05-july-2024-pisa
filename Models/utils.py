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
    df=pd.read_csv('../Dataset/resampled_dataset cluster centroids.csv')

    train_cols = df.columns.difference(['customer_types', 'conversion', 'offer'])
    X = df[train_cols] 
    y = df['treat']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=12)
    return X_train, X_test, Y_train, Y_test


def target_ova(df):
    #FIRST SPLIT (ONE VS ALL)
    #0(do not treat) -> 0,1,2
    #1(treat) -> 3

    """Declare target class
    """
    #Lost Causes:
    df.loc[df['customer_types'].isin([0, 1, 2]), 'treat'] = 0
    df.loc[df['customer_types'] == 3, 'treat'] = 1
    return df


def target_2(df):
    #SECOND SPLIT (TWO VS TWO)
    #0(do not treat) -> 0,1
    #1( treat) -> 2,3
    """Declare target class
    """
    #Lost Causes:
    df.loc[df['customer_types'].isin([0, 1]), 'treat'] = 0
    df.loc[df['customer_types'].isin([2,3]), 'treat'] = 1
    return df