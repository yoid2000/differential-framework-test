from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from joblib import dump, load
import numpy as np
import pandas as pd
import json
import pprint
import os
from mixed_types_kneighbors import MixedTypeKNeighbors

workWithAuto = False
if workWithAuto:
    import autosklearn.regression
    import autosklearn.classification

pp = pprint.PrettyPrinter(indent=4)

def getAnonymeterPreds(victims, dataset, secret, auxCols):
    ''' Both victims and dataset df's have all columns.
        The secret is the column the attacker is trying to predict
        In usage, the dataset can be the synthetic dataset (in which case
        we are emulating Anonymeter), or it can be the baseline dataset
        (in which case we are doing the differential framework, but with
        k-neighbors matching as our analysis)
        auxCols are the known columns
    '''
    print(f"getAnonymeterPreds for secret {secret}")
    secretType, nums, cats, drops = categorize_columns(dataset, secret)
    if secretType == 'drop':
        print(f"skip secret {secretType} because not cat or num")
        return
    print(f"Secret is {secret} with type {secretType}")
    for column in drops:
        victims = victims.drop(column, axis=1)
        dataset = dataset.drop(column, axis=1)
    nn = MixedTypeKNeighbors(n_neighbors=1).fit(candidates=dataset[auxCols])
    predictions_idx = nn.kneighbors(queries=victims[auxCols])
    predictions = dataset.iloc[predictions_idx.flatten()][secret]
    printEvaluation(secretType, victims[secret], predictions)

def doModel(fileBaseName, df, target, numVictims=500, auto='none'):
    if auto == 'autosklearn' and workWithAuto is False:
        return
    targetType, nums, cats, drops = categorize_columns(df, target)
    if targetType == 'drop':
        print(f"skip target {targetType} because not cat or num")
        return
    print(f"Target is {target} with type {targetType} and auto={auto}")
    for column in drops:
        df = df.drop(column, axis=1)

    # Assuming df is your DataFrame and 'target' is the column you want to predict
    X = df.drop(target, axis=1)
    y = df[target]


    if auto == 'none':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=numVictims, random_state=42)
        # Create a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), nums),
                ('cat', OneHotEncoder(), cats)
            ])

        # Create a pipeline that uses the transformer and then fits the model
        if targetType == 'cat':
            pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', LogisticRegression(penalty='l1', C=0.01, solver='saga'))])
        else:
            pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', Lasso(alpha=0.1))])

        # Fit the pipeline to the training data
        pipe.fit(X_train, y_train)

        # Use Logistic Regression with L1 penalty for feature selection and model building
        #model = LogisticRegression(penalty='l1', solver='liblinear')
        #model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = pipe.predict(X_test)
    elif auto == 'autosklearn':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=numVictims, random_state=42)
        if targetType == 'cat':
            # Initialize the classifier
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
            # Fit model
            automl.fit(X_train, y_train)
            # Print the final ensemble constructed by auto-sklearn
            print(automl.show_models())
            # Predict on test data
            y_pred = automl.predict(X_test)
        else:
            # Initialize the regressor
            automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
            # Fit model
            automl.fit(X_train, y_train)
            # Print the final ensemble constructed by auto-sklearn
            print(automl.show_models())
            # Predict on test data
            y_pred = automl.predict(X_test)
    elif auto == 'tpot':
        savedModelName = fileBaseName + '.tpot.joblib'
        if os.path.exists(savedModelName):
            tpot = load(savedModelName)
        else:
            for column in cats:
                df[column] = df[column].astype(str)
            X = pd.get_dummies(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=numVictims, random_state=42)
            if targetType == 'cat':
                # Initialize the classifier
                tpot = TPOTClassifier(generations=100, population_size=100, verbosity=2, random_state=42)
                # Fit the model
                tpot.fit(X_train, y_train)
                # Print the best pipeline
                print(tpot.fitted_pipeline_)
            else:
                # Initialize the regressor
                tpot = TPOTRegressor(generations=100, population_size=100, verbosity=2, random_state=42)
                # Fit the model
                tpot.fit(X_train, y_train)
                # Print the best pipeline
                print(tpot.fitted_pipeline_)
            dump(tpot.fitted_pipeline_, savedModelName)
        # Predict on test data
        y_pred = tpot.predict(X_test)

    printEvaluation(targetType, y_test, y_pred)

    # To see which features were selected
    if False and not auto:
        # Get the preprocessor step from the pipeline
        preprocessor = pipe.named_steps['preprocessor']

        # Get the feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()

        # Now use these feature names with your model coefficients
        selected_features = list(feature_names[(pipe.named_steps['model'].coef_ != 0).any(axis=0)])
        print(f"Selected features:")
        if targetType == 'cat':
            pp.pprint(list(selected_features))
            numFeatures = len(list(selected_features))
        else:
            print(type(selected_features))
            if len(selected_features) == 0:
                print("strange!!!")
                print(selected_features)
                numFeatures = 0
            else:
                pp.pprint(list(selected_features[0]))
                numFeatures = len(list(selected_features[0]))
        print(f"Selected {numFeatures} out of {len(feature_names)} total")

def printEvaluation(targetType, y_test, y_pred):
    if targetType == 'cat':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        # Find the most frequent category
        most_frequent = y_test.mode()[0]
        # Create a list of predictions
        y_pred_freq = [most_frequent] * len(y_test)
        # Compute accuracy
        accuracy_freq = accuracy_score(y_test, y_pred_freq)
        print(f"Accuracy of best guess: {accuracy_freq}")
        accuracy_improvement = (accuracy - accuracy_freq) / max(accuracy, accuracy_freq)
        print(f"Accuracy Improvement: {accuracy_improvement}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Average test value: {np.mean(y_test)}")
        print(f"Relative error: {rmse/np.mean(y_test)}")
        #npFixed = np.full((len(y_test),),0.001)
        #npMax = np.abs(np.maximum(y_test, y_pred, npFixed))
        #y_test_has_bad = not np.isfinite(y_test).all()
        #y_pred_has_bad = not np.isfinite(y_pred).all()
        #print(f"bad y_test {y_test_has_bad}, bad y_pred {y_pred_has_bad}")
        #relErr = np.abs(y_test - y_pred) / npMax
        #print(f"Average relative error: {np.mean(relErr)}")
        #print(f"Std Dev relative error: {np.std(relErr)}")

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

def categorize_columns(df, target):
    # Initialize empty lists for each category
    nums = []
    cats = []
    drops = []

    # Iterate over each column in the DataFrame except the target
    for col in df.columns:
        colType = getColType(df, col)
        if col == target:
            targetType = colType
            continue
        if colType == 'num':
            nums.append(col)
        if colType == 'cat':
            cats.append(col)
        if colType == 'drop':
            drops.append(col)
    return targetType, nums, cats, drops

def getColType(df, col):
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        if df[col].nunique() >= 10:
            return 'num'
        else:
            return 'cat'
    # Check if the column is object (string)
    elif pd.api.types.is_object_dtype(df[col]):
        if df[col].nunique() < 100:
            return 'cat'
        else:
            return 'drop'
    # If the column is neither numeric nor object, add it to 'drops'
    else:
        return 'drop'

    return nums, cats, drops


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

print_dataframe_columns(dfOrig)
print('===============================================')
print('===============================================')
for target in dfOrig.columns:
    print(f"Use target {target}")
    # Here, we are using the original rows to measure the baseline
    # using ML models (this is meant to give a high-quality baseline)
    fileBaseName = filePath + target
    print("----  DIFFERENTIAL FRAMEWORK  ----")
    doModel(fileBaseName, dfOrig, target, numVictims=numVictims)
    doModel(fileBaseName, dfOrig, target, auto='tpot', numVictims=numVictims)
    # Here, we are mimicing Anonymeter. That is to say, we are applying
    # the analysis (which is the same as the attack) to the synthetic
    # data using victims that were not part of making the synthetic data
    auxCols = list(dfTest.columns)
    auxCols.remove(target)
    print("----  CLASSIC ANONYMETER  ----")
    dfSample = dfTest.sample(n=numVictims, replace=False)
    getAnonymeterPreds(dfSample, dfAnon, target, auxCols)
    print('----------------------------------------------')