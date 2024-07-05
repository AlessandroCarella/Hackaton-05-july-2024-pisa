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
def getFeatureForClassification (df:pd.DataFrame):
    actualColumns = list(df.columns)
    
    out = []
    for column in featureForClassification:
        if column in actualColumns:
            out.append(column)
    return out

def plot_pca(X_pca, y_train):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
    plt.show()

def makeDecisionTreeClassifierMetrics (groundTruth, predictions):
    # Accuracy
    accuracy = accuracy_score(groundTruth, predictions)

    # Precision
    precision = precision_score(groundTruth, predictions, average='weighted', zero_division=0)

    # Recall
    recall = recall_score(groundTruth, predictions, average='macro', zero_division=0)

    # F1 Score
    f1 = f1_score(groundTruth, predictions, average='weighted')

    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1Score":f1
    }

def makeAndSaveToFileDecisionTreeResults(X_train, y_train, X_test, y_test, fileName, classWeight=None, adjustedPredict=False, adjustedPredictThreshold=None):
    """
    the output will be the created classifiers list and the results dictionary
    the results dictionary is also saved at the path given in the fileName
    the json is structured as follows
    criterion values
        max depth values
            min sample leafs values
                min sample split values
                    ccp alphas values
                        accuracy of the model
                        precision of the model
                        recall of the model
                        f1 score of the model

    btw None is the default value for the class_weight parameter, some balancing technique requires us to use this parameter so i'm
    defaulting the value to None and than the "balanced" string can be passed to the algh when needed                    
    
    if the adjustedPredict is setted to true the result structure is
    criterion values
        max depth values
            min sample leafs values
                min sample split values
                    ccp alphas values
                        threshold values
                            accuracy of the model
                            precision of the model
                            recall of the model
                            f1 score of the model

    """
    clfs = []
    results = {}
    
    criterions =  ['entropy', 'gini']
    maxDepths = list(np.arange(2, 3)) + [None]
    minSamplesLeaf = [ 0.1, 0.2, 1, 2 ,3 ,4 ,5 , 6]
    minSamplesSplit = [ 0.05, 0.1, 0.2]
    ccp_alphas = list(np.arange(0.01, 0.1))
    
    decisionTreeClassifierDict = {}
    for criterion in criterions:
        if str(criterion) not in results:
            results[criterion] = {}
        for maxDepth in maxDepths:
            if str(maxDepth) not in results[criterion]:
                results[criterion][str(maxDepth)] = {}
            for minSampleLeaf in minSamplesLeaf:
                if str(minSampleLeaf) not in results[criterion][str(maxDepth)]:
                    results[criterion][str(maxDepth)][str(minSampleLeaf)] = {}
                for minSampleSplit in minSamplesSplit:
                    if str(minSampleSplit) not in results[criterion][str(maxDepth)][str(minSampleLeaf)]:
                        results[criterion][str(maxDepth)][str(minSampleLeaf)][str(minSampleSplit)] = {}
                    for ccp_alpha in ccp_alphas:
                        
                        clf = DecisionTreeClassifier(
                                    max_depth=maxDepth, 
                                    criterion=criterion, 
                                    min_samples_leaf=minSampleLeaf, 
                                    min_samples_split=minSampleSplit, 
                                    ccp_alpha=ccp_alpha,
                                    class_weight=classWeight,
                                    random_state=42
                                )
                        clf.fit (X_train, y_train)
                        clfs.append (clf)

                        if adjustedPredict:
                            if str(ccp_alpha) not in results[criterion][str(maxDepth)][str(minSampleLeaf)][str(minSampleSplit)]:
                                results[criterion][str(maxDepth)][str(minSampleLeaf)][str(minSampleSplit)][str(ccp_alpha)] = {}
                            def adjusted_predict(X, thr=0.5):
                                y_score = clf.predict_proba(X_test)[:, 1]
                                return np.array([1 if y > thr else 0 for y in y_score])

                            y_pred = adjusted_predict(X_test, thr=adjustedPredictThreshold)

                            results[criterion][str(maxDepth)][str(minSampleLeaf)][str(minSampleSplit)][str(ccp_alpha)][str(adjustedPredictThreshold)] = makeDecisionTreeClassifierMetrics(y_test, y_pred)
                        else:
                            y_pred0 = clf.predict(X_test)

                            results[criterion][str(maxDepth)][str(minSampleLeaf)][str(minSampleSplit)][str(ccp_alpha)] = makeDecisionTreeClassifierMetrics(y_test, y_pred0)
                        
    with open (fileName + ".json", "w") as f:
        json.dump(results, f, indent=4)
     
    return clfs, results