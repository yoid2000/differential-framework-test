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
import pprint
import os
from mixed_types_kneighbors import MixedTypeKNeighbors

columnTypes = {
        'Attrition_Flag': 'cat',
        'Customer_Age': 'num',
        'Gender': 'cat',
        'Dependent_count': 'num',
        'Education_Level': 'cat',
        'Marital_Status': 'cat',
        'Income_Category': 'cat',
        'Card_Category': 'cat',
        'Months_on_book': 'num',
        'Total_Relationship_Count': 'num',
        'Months_Inactive_12_mon': 'num',
        'Contacts_Count_12_mon': 'num',
        'Credit_Limit': 'num',
        'Total_Revolving_Bal': 'num',
        'Avg_Open_To_Buy': 'num',
        'Total_Amt_Chng_Q4_Q1': 'num',
        'Total_Trans_Amt': 'num',
        'Total_Trans_Ct': 'num',
        'Total_Ct_Chng_Q4_Q1': 'num',
        'Avg_Utilization_Ratio': 'num',
}

workWithAuto = False
if workWithAuto:
    import autosklearn.regression
    import autosklearn.classification

pp = pprint.PrettyPrinter(indent=4)

def updateResults(res, dataset, column, measure, value):
    ''' measures are: 'accuracy', 'rmse', 'accuracy-freq', 'avg-value'
    '''
    if dataset not in res:
        res[dataset] = {}
    if column not in res[dataset]:
        res[dataset][column] = {}
    res[dataset][column][measure] = value

def getAnonymeterPreds(res, filePath, victims, dataset, secret, auxCols):
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
    printEvaluation(res, filePath, secret, secretType, victims[secret], predictions)

def doModel(res, dataset, target, df, numVictims=500, auto='none'):
    if auto == 'autosklearn' and workWithAuto is False:
        return
    targetType, nums, cats, drops = categorize_columns(df, target)
    if targetType == 'drop':
        print(f"skip target {targetType} because not cat or num")
        return
    print(f"Target is {target} with type {targetType} and auto={auto}")
    for column in drops:
        df = df.drop(column, axis=1)
    model, X_train, X_test, y_train, y_test = makeModel(dataset, target, df, numVictims=numVictims, auto=auto)
    y_pred = model.predict(X_test)
    if res is not None:
        printEvaluation(res, dataset, target, targetType, y_test, y_pred)


def makeModel(dataset, target, df, numVictims=500, auto='none', findLocal=False):
    fileBaseName = dataset + target
    targetType, nums, cats, drops = categorize_columns(df, target)
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

        return pipe, X_train, X_test, y_train, y_test
    elif auto == 'autosklearn':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=numVictims, random_state=42)
        if targetType == 'cat':
            # Initialize the classifier
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
            # Fit model
            automl.fit(X_train, y_train)
            # Print the final ensemble constructed by auto-sklearn
            print(automl.show_models())
            return automl, X_train, X_test, y_train, y_test
        else:
            # Initialize the regressor
            automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
            # Fit model
            automl.fit(X_train, y_train)
            # Print the final ensemble constructed by auto-sklearn
            print(automl.show_models())
            return automl, X_train, X_test, y_train, y_test
    elif auto == 'tpot':
        savedModelName = fileBaseName + '.tpot.joblib'
        if findLocal:
            savedModelPath = savedModelName
        else:
            savedModelPath = os.path.join('models', savedModelName)
        if os.path.exists(savedModelPath):
            tpot = load(savedModelPath)
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
            # This is supposed to be savedModelName...
            dump(tpot.fitted_pipeline_, savedModelName)
        # Predict on test data
        return tpot, X_train, X_test, y_train, y_test

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

def printEvaluation(res, dataset, target, targetType, y_test, y_pred):
    if targetType == 'cat':
        accuracy = accuracy_score(y_test, y_pred)
        updateResults(res, dataset, target, 'accuracy', accuracy)
        print(f"Accuracy: {accuracy}")
        # Find the most frequent category
        most_frequent = y_test.mode()[0]
        # Create a list of predictions
        y_pred_freq = [most_frequent] * len(y_test)
        # Compute accuracy
        accuracy_freq = accuracy_score(y_test, y_pred_freq)
        updateResults(res, dataset, target, 'accuracy_freq', accuracy_freq)
        print(f"Accuracy of best guess: {accuracy_freq}")
        accuracy_improvement = (accuracy - accuracy_freq) / max(accuracy, accuracy_freq)
        print(f"Accuracy Improvement: {accuracy_improvement}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        updateResults(res, dataset, target, 'rmse', rmse)
        updateResults(res, dataset, target, 'avg-value', np.mean(y_test))
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

if __name__ == "__main__":
    pass