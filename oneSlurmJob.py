import os
import sys
import fire
import pprint
import json
import pandas as pd
import diffTools
from joblib import dump

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run column quality measures on one test.
'''


def oneTpotJob(jobNum=0, jsonFile='BankChurnersNoId_ctgan.json', numVictims=500):
    with open(jsonFile, 'r') as f:
        testData = json.load(f)
    columns = testData['colNames']
    if jobNum >= len(columns):
        print(f"oneTpotJob: FAIL: jobNum {jobNum} exceeds columns {len(columns)}")
    dfOrig = pd.DataFrame(testData['originalTable'], columns=columns)
    target = columns[jobNum]
    fileBaseName = jsonFile + target
    print(f"Checking for {fileBaseName}")
    if os.path.exists(fileBaseName):
        print(f"{fileBaseName} already exists")
        print("oneTpotJob: SUCCESS")
        return
    model = diffTools.makeModel(fileBaseName, target, dfOrig, auto='tpot', numVictims=numVictims)
    # This is supposed to be savedModelName...
    dump(model.fitted_pipeline_, fileBaseName)
    print("oneTpotJob: SUCCESS")


def main():
    fire.Fire(oneTpotJob)


if __name__ == '__main__':
    main()
