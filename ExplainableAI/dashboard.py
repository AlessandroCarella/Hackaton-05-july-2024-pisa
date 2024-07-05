from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd 
import numpy as np
from utils import getDataset
from utils import get_names
import pickle

X_train, X_test, y_train, y_test = getDataset()
train_names, test_names = get_names()

df = pd.read_csv("../Dataset/final dataset.csv")

with open ("gmBoost.pickle", "rb") as f:
    model = pickle.load(f) 

model.fit(X_train, y_train)

feature_descriptions = {'channel': 'customer using, Phone/Web/Multichannel',
 'history': '$value of the historical purchases',
 'is_referral': 'indicates if the customer was acquired from referral channel',
 'recency': 'months since last purchase',
 'used_bogo': 'indicates if the customer used a buy one get one before',
 'used_discount': 'indicates if the customer used a discount before',
 'zip_code': 'class of the zip code as Suburban/Urban/Rural'}
explainer = ClassifierExplainer(model, X_test, y_test, 
                                    descriptions=feature_descriptions,
                                    idxs=test_names,
                                    target='treat', 
                                    labels=['non_treated','treated'])

ExplainerDashboard(explainer, mode='external').run()
