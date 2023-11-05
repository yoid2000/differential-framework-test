import os
import sys
import fire
import pprint
import json
import pandas as pd
import diffTest

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
    diffTest.doModel(fileBaseName, dfOrig, target, auto='tpot', numVictims=numVictims)
    print("oneTpotJob: SUCCESS")


def main():
    fire.Fire(oneTpotJob)


if __name__ == '__main__':
    main()
