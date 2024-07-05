#basic imports
import pandas as pd
import os
from os import path
import pickle
import json
import numpy as np
from tqdm import tqdm
import math

#teacher notebook imports

import seaborn as sns
import matplotlib.pyplot as plt

#other imports
#data partitioning
from sklearn.model_selection import train_test_split, cross_val_score

#classification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scikitplot.metrics import plot_roc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter
from collections import defaultdict
from sklearn.decomposition import PCA


featureForClassification = ['mode', 
 #'mode_confidence', 
 'energy', 'acousticness', 
 #'key_confidence', 
 'loudness', 'danceability', 
# 'tempo_confidence', 
'valence', 
#'time_signature_confidence', 
'speechiness', 'n_beats', 'time_signature', 
#'n_bars', 
#'start_of_fade_out', 'features_duration_ms', 'duration_ms', 'BPM', 'tempo', 'explicit', 'instrumentalness', 'artists_popularity_mean', 'artists_followers_mean', 'liveness', 'popularity'
]

featureForRegression = ['explicit', 'popularity', 
                        'danceability', 'energy', 
                        'mode', 'speechiness', 
                        'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'BPM']

def getFeatureForClassification (df:pd.DataFrame):
    actualColumns = list(df.columns)
    
    out = []
    for column in featureForClassification:
        if column in actualColumns:
            out.append(column)
    return out

def getFeatureForRegression (df:pd.DataFrame):
    actualColumns = list(df.columns)
    
    out = []
    for column in featureForRegression:
        if column in actualColumns:
            out.append(column)
    return out
