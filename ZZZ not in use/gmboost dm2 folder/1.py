import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from os import path
from datetime import datetime

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.special import expit

from utils import getFeatureForClassification

def findScoresMean (scores):
    meanAccuracy = 0
    meanPrecision = 0
    meanRecall = 0
    meanF1 = 0

    for score in scores:
        meanAccuracy += score["accuracy"]
        meanPrecision += score["precision"]
        meanRecall += score["recall"]
        meanF1 += score["f1Score"]

    return {
        "accuracy": meanAccuracy / len(scores),
        "precision": meanPrecision / len(scores),
        "recall": meanRecall / len(scores),
        "f1Score": meanF1 / len(scores),
    }

dfPath = r"C:\Users\alex1\Desktop\DataMining2\dataset\tabular\finalDataset.csv"

df = pd.read_csv(dfPath)

X = df[getFeatureForClassification(df)]
y = df["key"]

losses = ['log_loss', 'exponential']
learningRates = [0.001, 0.01, 0.1, 0.2]
n_esitmatorss = [50]
minSamplesSplitValues = [2, 4]
minSampleLeafValues = [1, 3]
maxDepths = [2, 3] + [None]
tolerances = [1e-2, 1e-4, 1e-6]


randomState = 42
nJobs = -1

holdout_percentages = [0.99, 0.1, 0.25, 0.5, 0.75]

for holdout_percentage in holdout_percentages:
    scl = StandardScaler()
    #X = scl.fit_transform(X)

    x_bad, X_holdout, y_bad, y_holdout = train_test_split(X, y, test_size=holdout_percentage, random_state=42, stratify=y)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    resultDict = {}
    for loss in losses:
        for learningRate in learningRates:
            for n_estimators in n_esitmatorss:
                for min_samples_split in minSamplesSplitValues:
                    for min_samples_leaf in minSampleLeafValues:
                        for max_depth in maxDepths:
                            for tolerance in tolerances:
                                print("Current hour:", datetime.now().hour, "Current second:", datetime.now().second)

                                key = " ".join(["loss:", str(loss), "learningRate:", str(learningRate), "n_estimator:", str(n_estimators), 
                                                "min_samples_split:", str(min_samples_split), "min_samples_leaf:", str(min_samples_leaf), "max_depth:", str(max_depth),
                                                "tolerance:", str(tolerance)])
                                print(key)

                                if key not in resultDict:
                                    clf = GradientBoostingClassifier(loss=loss, learning_rate=learningRate, 
                                                                     n_estimators=n_estimators, min_samples_split=min_samples_split, 
                                                                     min_samples_leaf=min_samples_leaf, max_depth=max_depth, 
                                                                     tol=tolerance, random_state=42)
                                    
                                    scores = []
                                    i = 0
                                    for train_index, test_index in kf.split(X_holdout):
                                        X_train, X_test = X_holdout.iloc[train_index], X_holdout.iloc[test_index]
                                        y_train, y_test = y_holdout.iloc[train_index], y_holdout.iloc[test_index]

                                        X_train_scaled = scl.fit_transform(X_train)
                                        X_test_scaled = scl.transform(X_test)

                                        clf.fit(X_train_scaled, y_train)
                                        y_pred = clf.predict(X_test_scaled)
                                        
                                        
                                        accuracy = accuracy_score(y_test, y_pred)
                                        # Precision
                                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                        # Recall
                                        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                                        # F1 Score
                                        f1 = f1_score(y_test, y_pred, average='weighted')
                                        scores.append(
                                            {
                                                "accuracy": accuracy,
                                                "precision": precision,
                                                "recall": recall,
                                                "f1Score": f1,
                                            }
                                        )

                                        i += 1

                                    #find the mean of all the values
                                    scoresMean = findScoresMean(scores)

                                    resultDict[key] = scoresMean

        with open("Gradient boost classifier resultDict holdout " + str(1-holdout_percentage) + ".json", "w") as f:
            json.dump(resultDict, f, indent=4)

        print("Current hour:", datetime.now().hour, "Current second:", datetime.now().second)
