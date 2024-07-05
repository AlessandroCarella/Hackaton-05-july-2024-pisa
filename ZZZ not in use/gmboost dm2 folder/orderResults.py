import json
from os import path
import os

folderPath = path.join (os.getcwd(), "src", "Classification", "Tabular", "Gradient Boosting Machines")
results = {}

for file in os.listdir(folderPath):
    filePath = path.join (folderPath, file)
    filename, ext = os.path.splitext (filePath)
    if "json" in ext and "aggregated" not in filename:
        with open(filePath, "r") as f:
            results[path.basename(filename).replace("Gradient boosting classifier resultDict ", "")] = json.load(f)


initialKeys = []
for k, v in results.items():
    for key, value in v.items():
        initialKeys.append(key)

for result in results:
    sorted_models = sorted(results[result].items(), key=lambda x: (x[1]["accuracy"], x[1]["precision"], x[1]["recall"], x[1]["f1Score"]), reverse=True)
    results[result] = dict(sorted_models)

newKeys = []
for k, v in results.items():
    for key, value in v.items():
        newKeys.append(key)

print (newKeys == initialKeys) #False, the keys are ordered

with open (path.join (folderPath, "aggregatedResults.json"), "w") as f:
    json.dump(results, f, indent=4)

