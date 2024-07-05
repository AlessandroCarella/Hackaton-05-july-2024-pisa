import numpy as np
import pandas as pd
import json
import datetime
import multiprocessing

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import getFeatureForClassification

def findScoresMean(scores):
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

def process_parameter(loss, learningRate, n_estimators, min_samples_split, min_samples_leaf, max_depth, tolerance, X_holdout, y_holdout, kf, scl, key):
    print ("START Processing", key, datetime.datetime.now().strftime("%A, %H:%M:%S"))
    
    scores = []
    for train_index, test_index in kf.split(X_holdout):
        X_train, X_test = X_holdout.iloc[train_index], X_holdout.iloc[test_index]
        y_train, y_test = y_holdout.iloc[train_index], y_holdout.iloc[test_index]

        X_train_scaled = scl.fit_transform(X_train)
        X_test_scaled = scl.transform(X_test)

        clf = GradientBoostingClassifier(loss=loss, learning_rate=learningRate, 
                                         n_estimators=n_estimators, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, max_depth=max_depth, 
                                         tol=tolerance, random_state=42)

        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')

        scores.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1Score": f1,
        })

    print ("END Processing", key, datetime.datetime.now().strftime("%A, %H:%M:%S"))
    
    return scores

if __name__ == '__main__':
    dfPath = r"C:\Users\alex1\Desktop\DataMining2\dataset\tabular\finalDataset.csv"
    df = pd.read_csv(dfPath)
    X = df[getFeatureForClassification(df)]
    y = df["key"]

    losses = ['log_loss']#, 'exponential']
    learningRates = [#0.001, 
                     0.01, 0.1, 0.2]
    n_esitmatorss = [25, 50]
    minSamplesSplitValues = [2, 4]
    minSampleLeafValues = [1, 3]
    maxDepths = [3, 2]
    tolerances = [1e-3, 1e-4]#, 1e-6]

    randomState = 42

    holdout_percentages = [0.99, 0.1, 0.25, 0.5, 0.75]

    for holdout_percentage in holdout_percentages:
        scl = StandardScaler()
        # X = scl.fit_transform(X)

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=holdout_percentage, random_state=randomState, stratify=y)

        kf = KFold(n_splits=10, shuffle=True, random_state=randomState)

        results = {}
        jobs = []

        for loss in losses:
            for learningRate in learningRates:
                for n_estimators in n_esitmatorss:
                    for min_samples_split in minSamplesSplitValues:
                        for min_samples_leaf in minSampleLeafValues:
                            for max_depth in maxDepths:
                                for tolerance in tolerances:
                                    key = " ".join(["loss:", str(loss), "learningRate:", str(learningRate), "n_estimator:", str(n_estimators), 
                                                    "min_samples_split:", str(min_samples_split), "min_samples_leaf:", str(min_samples_leaf), "max_depth:", str(max_depth),
                                                    "tolerance:", str(tolerance)])

                                    if key not in results:
                                        jobs.append((loss, learningRate, n_estimators, min_samples_split, min_samples_leaf, max_depth, tolerance, X_holdout, y_holdout, kf, scl, key))

        with multiprocessing.Pool(processes=int((multiprocessing.cpu_count()/2))) as pool:
            scores = pool.starmap(process_parameter, jobs)

        for job, score in zip(jobs, scores):
            key = " ".join(["loss:", str(job[0]), "learningRate:", str(job[1]), "n_estimator:", str(job[2]), 
                            "min_samples_split:", str(job[3]), "min_samples_leaf:", str(job[4]), "max_depth:", str(job[5]),
                            "tolerance:", str(job[6])])
            results[key] = findScoresMean(score)

        with open("Gradient boosting classifier resultDict holdout {}.json".format(1-holdout_percentage), "w") as f:
            json.dump(results, f, indent=4)
