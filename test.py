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
import numpy as np
import pandas as pd
import pprint

workWithAuto = False
if workWithAuto:
    import autosklearn.regression
    import autosklearn.classification

pp = pprint.PrettyPrinter(indent=4)


def doModel(df, target, auto='none'):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        for column in cats:
            df[column] = df[column].astype(str)
        X = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if targetType == 'cat':
            # Initialize the classifier
            tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
            # Fit the model
            tpot.fit(X_train, y_train)
            # Print the best pipeline
            print(tpot.fitted_pipeline_)
            # Predict on test data
            y_pred = tpot.predict(X_test)
        else:
            # Initialize the regressor
            tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
            # Fit the model
            tpot.fit(X_train, y_train)
            # Print the best pipeline
            print(tpot.fitted_pipeline_)
            # Predict on test data
            y_pred = tpot.predict(X_test)


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


# Usage
file_path = 'BankChurnersNoId.csv'  # replace with your file path
df = csv_to_dataframe(file_path)
print_dataframe_columns(df)
print('===============================================')
print('===============================================')
for target in df.columns:
    print(f"Use target {target}")
    doModel(df, target)
    doModel(df, target, auto='tpot')
    print('----------------------------------------------')