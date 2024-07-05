import json
from os import path
import os

folderPath = path.join (os.getcwd(), "src", "Classification", "Tabular", "Gradient Boosting Machines")

with open (path.join (folderPath, "aggregatedResults.json")) as f:
    results = json.load(f)

bestValues = {}
for file in results:
    bestValues[file] = {}
    bestValues[file][list(results[file].keys())[0]] = results[file][list(results[file].keys())[0]]

for k, v in bestValues.items():
    print (k, v)
    print()