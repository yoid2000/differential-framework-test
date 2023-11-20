from diffTools import makeModel, getAnonymeterPreds, categorize_columns, printEvaluation, prepDataframes, StoreResults
import pandas as pd
import numpy as np
import json
import fire
import pprint

workWithAuto = False
pp = pprint.PrettyPrinter(indent=4)

def doModel(sr, method, dataset, target, df, dfTest, auto='none', max_iter=100, doPerValuePrecision=False):
    if auto == 'autosklearn' and workWithAuto is False:
        return
    targetType, nums, cats, drops = categorize_columns(df, target)
    if targetType == 'drop':
        print(f"skip target {targetType} because not cat or num")
        return
    print(f"Target is {target} with type {targetType} and auto={auto}")
    for column in drops:
        df = df.drop(column, axis=1)
        dfTest = df.drop(column, axis=1)
    X_test = dfTest.drop(target, axis=1)
    y_test = dfTest[target]
    model = makeModel(dataset, target, df, auto=auto, max_iter=max_iter)
    y_pred = model.predict(X_test)
    printEvaluation(sr, method, dataset, target, targetType, y_test, y_pred, doBestGuess=True, doPerValuePrecision=doPerValuePrecision)

def sample_rows(df, num_rows=500):
    # Randomly sample rows and create a new dataframe
    sampled_df = df.sample(n=num_rows)
    # Remove the sampled rows from the original dataframe
    df = df.drop(sampled_df.index)
    return df, sampled_df

def replicate_rows(df, frac = 0.1):
    # Calculate the number of rows to replicate
    num_rows = int(len(df) * frac)
    # Randomly select rows
    replicate_rows = df.sample(n=num_rows)
    # Append the replicated rows to the dataframe
    df = pd.concat([df, replicate_rows], ignore_index=True)
    return df

def csv_to_dataframe(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Return the DataFrame
    return df

def print_dataframe_columns(df):
    # Loop through each column
    for column in df.columns:
        # Print the column description
        print("-----")
        print(df[column].describe())

def runDiffTest(jobNum=0, numVictims = None, max_iter = 5000, doTpot = False, resultsFile = 'results.json', filePath = 'BankChurnersNoId_ctgan.json', doReplication=False, doNonMemberOnly=False, doPerValuePrecision=True):
    sr = StoreResults(resultsFile)
    with open(filePath, 'r') as f:
        testData = json.load(f)
    '''
        dfAnon is the synthetic data generated from dfOrig
        dfTest is additional data not used for the synthetic data
    '''
    dfOrig = pd.DataFrame(testData['originalTable'], columns=testData['colNames'])
    dfAnon = pd.DataFrame(testData['anonTable'], columns=testData['colNames'])
    dfTest = pd.DataFrame(testData['testTable'], columns=testData['colNames'])

    dfOrig, dfTest, dfAnon = prepDataframes(dfOrig, dfTest, dfAnon)

    print(f"Got {dfOrig.shape[0]} original rows")
    print(f"Got {dfAnon.shape[0]} synthetic rows")
    print(f"Got {dfTest.shape[0]} test rows")

    print(list(dfOrig.columns))

    if numVictims is None:
        numVictims = dfTest.shape[0]
    dfSampleVictims = dfTest.sample(n=numVictims, replace=False)
    if True:
        ''' The following runs the tests for the case where the victims are
            completely distinct from the original data
        '''
        
        print_dataframe_columns(dfOrig)
        print('===============================================')
        print('===============================================')
        for target in dfOrig.columns:
            print(f"Use target {target}")
            # Here, we are using the original rows to measure the baseline
            # using ML models (this is meant to give a high-quality baseline)
            print("\n----  DIFFERENTIAL FRAMEWORK  ----")
            doModel(sr, 'manual-ml', filePath, target, dfOrig, dfSampleVictims, max_iter=max_iter, doPerValuePrecision=doPerValuePrecision)
            if doTpot:
                doModel(sr, 'tpot', filePath, target, dfOrig, dfSampleVictims, auto='tpot')
            if not doNonMemberOnly:
                # Here, we are mimicing Anonymeter. That is to say, we are applying
                # the analysis (which is the same as the attack) to the synthetic
                # data using victims that were not part of making the synthetic data
                auxCols = list(dfTest.columns)
                auxCols.remove(target)
                print("\n----  CLASSIC ANONYMETER  ----")
                getAnonymeterPreds(sr, 'classic-anonymeter', filePath, dfSampleVictims, dfAnon, target, auxCols)
                # Here, we are running Anonymeter against the original data instead of
                # the synthetic data. This follows the differential framework, but
                # using Anonymeter's attack method as the analysis
                print('----------------------------------------------')
                print("\n----  DIFFERENTIAL ANONYMETER  ----")
                getAnonymeterPreds(sr, 'diff-anonymeter', filePath, dfSampleVictims, dfOrig, target, auxCols)

    if doReplication:
        ''' The following runs the tests for the case where some fraction of the
            victims are replicated in the original data. The idea here is to model
            the case where some users are linked (i.e. husband and wife that often
            travel together), and one linked user is among the victims while the
            other is among the original data.
        '''
        linkages = [10, 50, 100]      # these are precentages

        for linkage in linkages:
            dfOrigLinked = replicate_rows(dfOrig, frac= (linkage/100))
            dfOrigLinked, dfTestLinked = sample_rows(dfOrigLinked, num_rows=numVictims)
            print(f"Len orig: {dfOrig.shape[0]}")
            print(f"Len linked: {dfOrigLinked.shape[0]}")
            print(f"Len test linked: {dfTestLinked.shape[0]}")
            for target in dfOrigLinked.columns:
                print('----------------------------------------------')
                print(f"Use target {target}")
                # Here, we are using the original rows to measure the baseline
                # using ML models (this is meant to give a high-quality baseline)
                print(f"\n----  DIFFERENTIAL FRAMEWORK  ({linkage}%)----")
                doModel(sr, f'manual-ml-{linkage}', filePath, target, dfOrigLinked, dfTestLinked, max_iter=max_iter)
                auxCols = list(dfTest.columns)
                auxCols.remove(target)
                # Here, we are running Anonymeter against the original data instead of
                # the synthetic data. This follows the differential framework, but
                # using Anonymeter's attack method as the analysis
                print(f"\n----  DIFFERENTIAL ANONYMETER  ({linkage}%)----")
                getAnonymeterPreds(sr, f'diff-anonymeter-{linkage}', filePath, dfTestLinked, dfOrigLinked, target, auxCols)

def main():
    fire.Fire(runDiffTest)

if __name__ == '__main__':
    main()