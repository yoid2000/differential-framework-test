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
from joblib import load
import numpy as np
import pandas as pd
import pprint
import json
import time
import os
from mixed_types_kneighbors import MixedTypeKNeighbors

columnTypes = {
        'Attrition_Flag': 'cat',
        'Customer_Age': 'num',
        'Gender': 'cat',
        'Dependent_count': 'num',    #num
        'Education_Level': 'cat',
        'Marital_Status': 'cat',
        'Income_Category': 'cat',
        'Card_Category': 'cat',
        'Months_on_book': 'num',
        'Total_Relationship_Count': 'num',
        'Months_Inactive_12_mon': 'num',    #num
        'Contacts_Count_12_mon': 'num',     #num
        'Credit_Limit': 'num',
        'Total_Revolving_Bal': 'num',
        'Avg_Open_To_Buy': 'num',
        'Total_Amt_Chng_Q4_Q1': 'num',
        'Total_Trans_Amt': 'num',
        'Total_Trans_Ct': 'num',
        'Total_Ct_Chng_Q4_Q1': 'num',
        'Avg_Utilization_Ratio': 'num',
}

class StoreResults():
    def __init__(self, resultsFileName):
        os.makedirs('results', exist_ok=True)
        self.resultsFileName = os.path.join('results', resultsFileName)

    def updateResults(self, method, dataset, column, measure, value):
        if not os.path.exists(self.resultsFileName):
            print(f"updateResults: {self.resultsFileName} doesn't exist: making")
            res = {}
        else:
            print(f"updateResults: opening {self.resultsFileName}")
            with open(self.resultsFileName, 'r') as f:
                res = json.load(f)
        if method not in res:
            res[method] = {}
        if dataset not in res:
            res[method][dataset] = {}
        if column not in res[method][dataset]:
            res[method][dataset][column] = {}
        measureList = measure + '__'
        if measureList not in res[method][dataset][column]:
            res[method][dataset][column][measureList] = []
        res[method][dataset][column][measureList].append(value)
        res[method][dataset][column][measure] = sum(res[method][dataset][column][measureList]) / len(res[method][dataset][column][measureList])
        print(f"updateResults: writing {self.resultsFileName}")
        with open(self.resultsFileName, 'w') as f:
            json.dump(res, f, indent=4)


pp = pprint.PrettyPrinter(indent=4)

def getPrecisionFromBestGuess(y_test):
    # Find the most frequent category
    most_frequent = y_test.mode()[0]
    print(f"most_frequent is {most_frequent}")
    most_frequent_count = y_test.value_counts()
    most_frequent_count = (y_test == most_frequent).sum()
    print(f"most_frequent_count {most_frequent_count}")
    return(most_frequent_count / len(y_test))

def convert_to_numpy(var):
    if isinstance(var, pd.Series):
        return var.values
    elif isinstance(var, np.ndarray):
        return var
    else:
        print("The input is neither a pandas Series nor a numpy array.")
        return None

def doClip(thingy,clipBegin = 10, clipEnd=3):
    clipped = []
    for thing in thingy:
        clip = thing[:clipBegin] + '.' + thing[-clipEnd:]
        while clip in clipped:
            clip += '_'
        clipped.append(clip)
    return clipped

def printEvaluation(sr, method, dataset, target, targetType, y_test, y_pred, precError=0.05, doBestGuess=True):
    if targetType == 'cat':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        accuracy_freq = getPrecisionFromBestGuess(y_test)
        if doBestGuess:
            accuracy = max(accuracy, accuracy_freq)
        sr.updateResults(method, dataset, target, 'accuracy', accuracy)
        sr.updateResults(method, dataset, target, 'accuracy_freq', accuracy_freq)
        print(f"Accuracy of best guess: {accuracy_freq}")
        accuracy_improvement = (accuracy - accuracy_freq) / max(accuracy, accuracy_freq)
        print(f"Accuracy Improvement: {accuracy_improvement}")
    else:
        # First compute rmse
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        sr.updateResults(method, dataset, target, 'rmse', rmse)
        sr.updateResults(method, dataset, target, 'avg-value', np.mean(y_test))
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Average test value: {np.mean(y_test)}")
        print(f"Relative error: {rmse/np.mean(y_test)}")
        # Then compute precision of prediction within some error tolerance
        testRange = abs(max(y_test) - min(y_test))
        errorTolorance = precError * testRange
        lowEdges = y_test - errorTolorance
        highEdges = y_test + errorTolorance
        print(f"testRange: {testRange}, errorTolorance: {errorTolorance}")
        print("y_test")
        print(y_test)
        print("y_pred")
        print(y_pred)
        print(lowEdges)
        print(highEdges)
        print('types', type(y_pred), type(lowEdges), type(highEdges))
        correctPredictions = ((convert_to_numpy(y_pred) >= convert_to_numpy(lowEdges)) & (convert_to_numpy(y_pred) <= convert_to_numpy(highEdges)))
        print("correctPredictions")
        print(correctPredictions)
        numCorrect = np.count_nonzero(correctPredictions)
        errorPrecision = numCorrect / len(y_test)
        if doBestGuess:
            errorPrecision_freq = getPrecisionFromBestGuess(y_test)
            errorPrecision = max(errorPrecision, errorPrecision_freq)
        # Now we check to see if we could have gotten a better prediction by
        # simply predicting the most frequent value
        sr.updateResults(method, dataset, target, 'errorPrecision', errorPrecision)

def getAnonymeterPreds(sr, method, filePath, victims, dataset, secret, auxCols):
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
    printEvaluation(sr, method, filePath, secret, secretType, victims[secret], predictions)

def makeModel(dataset, target, df, auto='none', max_iter=100):
    fileBaseName = dataset + target
    targetType, nums, cats, drops = categorize_columns(df, target)
    # Assuming df is your DataFrame and 'target' is the column you want to predict
    X = df.drop(target, axis=1)
    y = df[target]

    if auto == 'none':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)
        # Create a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), nums),
                ('cat', OneHotEncoder(), cats)
            ])

        # Create a pipeline that uses the transformer and then fits the model
        if targetType == 'cat':
            pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', LogisticRegression(penalty='l1', C=0.01, solver='saga', max_iter=max_iter))])
        else:
            pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', Lasso(alpha=0.1))])

        # Fit the pipeline to the training data
        pipe.fit(X_train, y_train)

        # Use Logistic Regression with L1 penalty for feature selection and model building
        #model = LogisticRegression(penalty='l1', solver='liblinear')
        #model.fit(X_train, y_train)

        return pipe
    elif auto == 'autosklearn':
        import autosklearn.regression
        import autosklearn.classification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)
        if targetType == 'cat':
            # Initialize the classifier
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
            # Fit model
            automl.fit(X_train, y_train)
            # Print the final ensemble constructed by auto-sklearn
            print(automl.show_models())
            return automl
        else:
            # Initialize the regressor
            automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
            # Fit model
            automl.fit(X_train, y_train)
            # Print the final ensemble constructed by auto-sklearn
            print(automl.show_models())
            return automl
    elif auto == 'tpot':
        savedModelName = fileBaseName + '.tpot.joblib'
        savedModelPath = os.path.join('models', savedModelName)
        if os.path.exists(savedModelPath):
            tpot = load(savedModelPath)
            return tpot
        else:
            for column in cats:
                df[column] = df[column].astype(str)
            X = pd.get_dummies(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)
            if targetType == 'cat':
                # Initialize the classifier
                tpot = TPOTClassifier(generations=100, population_size=100, verbosity=1)
                # Fit the model
                tpot.fit(X_train, y_train)
                # Print the best pipeline
                print(tpot.fitted_pipeline_)
            else:
                # Initialize the regressor
                tpot = TPOTRegressor(generations=100, population_size=100, verbosity=1)
                # Fit the model
                tpot.fit(X_train, y_train)
                # Print the best pipeline
                print(tpot.fitted_pipeline_)
        # Predict on test data
        return tpot

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

def categorize_columns(df, target):
    # Initialize empty lists for each category
    nums = []
    cats = []
    drops = []

    # Iterate over each column in the DataFrame except the target
    targetType = None
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
    if col in columnTypes:
        return columnTypes[col]
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

def prepDataframes(dfOrig, dfTest, dfAnon):
    _, nums, cats, drops = categorize_columns(dfOrig, 'none')
    for column in drops:
        dfOrig = dfOrig.drop(column, axis=1)
        dfTest = dfTest.drop(column, axis=1)
        dfAnon = dfAnon.drop(column, axis=1)
    columns = list(dfOrig.columns)
    for column in columns:
        if column[:5] == 'Naive':
            dfOrig = dfOrig.drop(column, axis=1)
            dfAnon = dfAnon.drop(column, axis=1)
            dfTest = dfTest.drop(column, axis=1)
    return dfOrig, dfTest, dfAnon

if __name__ == "__main__":
    pass