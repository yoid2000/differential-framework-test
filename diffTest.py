from diffTools import doModel, getAnonymeterPreds 
import pandas as pd
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)

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

if __name__ == "__main__":
    results = {
        'tpot':{},
        'manual-ml':{},
        'classic-anonymeter':{},
        'diff-anonymeter':{},
        'manual-ml-10':{},
        'manual-ml-50':{},
        'manual-ml-100':{},
        'diff-anonymeter-10':{},
        'diff-anonymeter-50':{},
        'diff-anonymeter-100':{},
    }
    doTpot = False
    numVictims = 500
    filePath = 'BankChurnersNoId_ctgan.json'
    with open(filePath, 'r') as f:
        testData = json.load(f)
    '''
        dfAnon is the synthetic data generated from dfOrig
        dfTest is additional data not used for the synthetic data
    '''
    dfOrig = pd.DataFrame(testData['originalTable'], columns=testData['colNames'])
    dfAnon = pd.DataFrame(testData['anonTable'], columns=testData['colNames'])
    dfTest = pd.DataFrame(testData['testTable'], columns=testData['colNames'])

    print(f"Got {dfOrig.shape[0]} original rows")
    print(f"Got {dfAnon.shape[0]} synthetic rows")
    print(f"Got {dfTest.shape[0]} test rows")

    # I want to clean out the two "Naive_Bayes..." columns, because they are
    # not original data
    columns = list(dfOrig.columns)
    for column in columns:
        if column[:5] == 'Naive':
            dfOrig = dfOrig.drop(column, axis=1)
            dfAnon = dfAnon.drop(column, axis=1)
            dfTest = dfTest.drop(column, axis=1)
    print(list(dfOrig.columns))

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
            doModel(results['manual-ml'], filePath, target, dfOrig, numVictims=numVictims)
            if doTpot:
                doModel(results['tpot'], filePath, target, dfOrig, auto='tpot', numVictims=numVictims)
            # Here, we are mimicing Anonymeter. That is to say, we are applying
            # the analysis (which is the same as the attack) to the synthetic
            # data using victims that were not part of making the synthetic data
            auxCols = list(dfTest.columns)
            auxCols.remove(target)
            dfSampleVictims = dfTest.sample(n=numVictims, replace=False)
            print("\n----  CLASSIC ANONYMETER  ----")
            getAnonymeterPreds(results['classic-anonymeter'], filePath, dfSampleVictims, dfAnon, target, auxCols)
            # Here, we are running Anonymeter against the original data instead of
            # the synthetic data. This follows the differential framework, but
            # using Anonymeter's attack method as the analysis
            print('----------------------------------------------')
            print("\n----  DIFFERENTIAL ANONYMETER  ----")
            getAnonymeterPreds(results['diff-anonymeter'], filePath, dfSampleVictims, dfOrig, target, auxCols)
        pp.pprint(results)
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
        

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
            doModel(results[f'manual-ml-{linkage}'], filePath, target, dfOrigLinked, numVictims=numVictims)
            auxCols = list(dfTest.columns)
            auxCols.remove(target)
            dfSampleVictims = dfTestLinked.sample(n=numVictims, replace=False)
            # Here, we are running Anonymeter against the original data instead of
            # the synthetic data. This follows the differential framework, but
            # using Anonymeter's attack method as the analysis
            print(f"\n----  DIFFERENTIAL ANONYMETER  ({linkage}%)----")
            getAnonymeterPreds(results[f'diff-anonymeter-{linkage}'], filePath, dfSampleVictims, dfOrigLinked, target, auxCols)
        pp.pprint(results)
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)