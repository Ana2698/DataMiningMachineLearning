from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def save_kfold(kfold_scores: pd.DataFrame, mnist: bool) -> None:
    COLS = [*[f"fold{i}" for i in range(1, 6)], "mean"]
    INDEX = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]
    # name checks
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to "
                         "`save` must be a pandas DataFrame.")
    if kfold_scores.shape != (6, 6):
        raise ValueError("DataFrame must "
                         "have 6 rows and 6 columns.")
    if df.columns.to_list() != COLS:
        raise ValueError(
            f"Columns are incorrectly named and"
            f"/or incorrectly sorted. "
            f"Got:\n{df.columns.to_list()}\n"
            f"but expected:\n{COLS}"
        )
    if df.index.name.lower() != "classifier":
        raise ValueError(
            "Index is incorrectly named. "
            "Create a proper index using `pd.Series` or "
            "`pd.Index` and use the "
            "`name='classifier'` argument."
        )
    if sorted(df.index.to_list()) != sorted(INDEX):
        raise ValueError(f"Incorrect names for index "
                         f"values. Index values must be {INDEX} ")
    # value range checks
    if mnist:
        if np.min(df.values) < 0 or np.max(df.values) > 0.10:
            raise ValueError(
                "Your K-Fold MNIST error rates are too extreme. "
                "Ensure they are the raw error rates,\r\n"
                "and NOT percentage error rates. "
                "Also ensure your DataFrame contains error rates,\r\n"
                "and not accuracies. If you are sure you "
                "have not made either of the above mistakes,\r\n"
                "there is probably something else wrong with your code. "
                "Contact the TA for help.\r\n"
            )
        if df.loc["svm_linear", "mean"] > 0.07:
            raise ValueError("Your svm_linear error rate is too high. "
                             "There is likely an error in your code.")
        if df.loc["svm_rbf", "mean"] > 0.03:
            raise ValueError("Your svm_rbf error rate is too high. "
                             "There is likely an error in your code.")
        if df.loc["rf", "mean"] > 0.05:
            raise ValueError("Your Random Forest error rate is too high. "
                             "There is likely an error in your code.")
        if df.loc[["knn1", "knn5", "knn10"], "mean"].min() > 0.04:
            raise ValueError("One of your KNN error rates is too high. "
                             "There is likely an error in your code.")
        outfile = Path("kfold_mnist.json")
    else:
        outfile = Path("kfold_data.json")
    df.to_json(outfile)
    print(f"K-Fold error rates for {'MNIST ' if mnist else ''}"
          f"data successfully saved to {outfile}")


"""
model_scores() is used to split the data into 5 folds for cross validation,
trains and compares 6 classifiers.
It takes X and y as input, returns a list of test errors.
"""


def model_scores(X, y):
    # The "models" dictionary defines all the classifier being compared.
    models = {
        'svm_linear': SVC(kernel='linear', random_state=0),
        'svm_rbf': SVC(kernel='rbf', random_state=0, ),
        'rf': RF(n_estimators=100, random_state=0),
        'knn1': KNeighborsClassifier(n_neighbors=1),
        'knn5': KNeighborsClassifier(n_neighbors=5),
        'knn10': KNeighborsClassifier(n_neighbors=10)
    }
    model_testerror = []
    test_error = []
    # Kfold splits n_splits=5
    skf = StratifiedKFold(n_splits=5, random_state=None)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # classifier dictionary is iterated over
        for key, value in models.items():
            value.fit(X_train, y_train.values.ravel())
            prediction = value.predict(X_test)
            # The test errors for each classifier is
            # calculated and rounded to 3 decimal points,
            # test errors are stored in model_testerror list
            model_testerror.append((1 -
                                    accuracy_score(y_test, prediction)
                                    ))
        # The "test_error" list stores test_errors for all 6 classifiers,
        # over all 5 folds
        test_error.append(model_testerror)
        model_testerror = []
    return test_error


"""
model_scoredf() is used to create the dataframe that contains
the K-Fold error rates for each classifier.
It takes list of test errors return by the model_scores() functions.
And it returns the required dataframe.
"""


def model_scoredf(score_list):
    classifier_names = ['svm_linear', 'svm_rbf', 'rf', 'knn1', 'knn5', 'knn10']
    # Dictionary containing the dataframe's column names as key
    # and the dataframe's row values as value.
    df_dict = {'fold1': score_list[0],
               'fold2': score_list[1],
               'fold3': score_list[2],
               'fold4': score_list[3],
               'fold5': score_list[4]
               }
    # Dataframe created with the classifier names as the index
    df = pd.DataFrame(df_dict, index=classifier_names)
    df.index.name = 'classifier'
    # Mean of the test errors of all 5 folds for each model is calculated,
    # and stored in a new column in the dataframe created above.
    df['mean'] = df.mean(axis=1, numeric_only=True)
    return df


def function1(path):
    # Loading given mnist data from the file
    num_data1 = loadmat(path)
    # X with shape (28, 28, 3000), y with shape (1, 3000)
    # X has the features and samples,
    # y contains labels of the samples
    x_data, y_data = num_data1['X'], num_data1['y']
    # x_data with shape (3000, 28, 28), y_data with shape( 3000, 1)
    x_data, y_data = x_data.transpose(2, 0, 1), y_data.transpose(1, 0)
    # Reshaping x_data to the shape (3000, 784)
    x_samples, x_features = x_data.shape[0], np.prod(x_data.shape[1:])
    x_data = x_data.reshape(x_samples, x_features)
    # Recording indices of labels <=7 in y_data array in index.
    index = np.where(y_data <= 7)[0]
    index = index.tolist()
    # Removing the recorded indices leaving only 8s and 9s in y_data.
    y_data = np.delete(y_data, index)
    y, x_data = pd.DataFrame(y_data), pd.DataFrame(x_data)
    # Removing samples of <=7 samples from x_data,
    # by matching the indices recorded in index.
    X = x_data.drop(index, axis=0)
    # model_score function called to split the data, calculate test_errors.
    test_errors = model_scores(X, y)
    # model_scoredf called to create the required dataframe.
    df = model_scoredf(test_errors)
    # save_kfold called to create the required json file.
    save_kfold(df, mnist=True)


def function2(path):
    auc = []
    columns = []
    df = pd.read_csv(path)
    # Mapping categorical data to numerical values
    df['Industry'].replace(
        {'Industrials': 0, 'Materials': 1,
         'CommunicationServices': 2, 'Transport': 3,
         'InformationTechnology': 4, 'Financials': 5,
         'Energy': 6, 'Real Estate': 7,
         'Utilities': 8, 'ConsumerDiscretionary': 9,
         'Education': 10, 'ConsumerStaples': 11,
         'Healthcare': 12, 'Research': 13},
        inplace=True)
    df['Ethnicity'].replace({'White': 0,
                             'Black': 1,
                             'Asian': 2,
                             'Latino': 3,
                             'Other': 4},
                            inplace=True)
    df['Citizen'].replace({'ByBirth': 0,
                           'ByOtherMeans': 1,
                           'Temporary': 2},
                          inplace=True)
    # Separating X (samples, features) and y (labels) from dataframe.
    X = df.drop(['Approved'], axis=1)
    y = df['Approved']
    for i in X.columns:
        features = np.array(X[i])
        # AUC values for each feature calculated
        if (roc_auc_score(y, features)) < 0.5:
            auc.append(1 - (roc_auc_score(y, features)))
        else:
            auc.append(roc_auc_score(y, features))
        columns.append(i)
    # AUC values stored in a Dataframe.
    auc_measurements = pd.DataFrame(list(zip(columns, auc)),
                                    columns=['feature', 'auc'])
    # Sorting the AUC values in descending order.
    auc_measurements.sort_values(by='auc', ascending=False,
                                 inplace=True)
    # Dropping index column from the dataframe.
    auc_measurements.reset_index(drop=True, inplace=True)

    # Code taken from the instructions provided.
    # Function used to create auc.json file.
    def validate_aucs(aucs: DataFrame) -> None:
        aucs = aucs.rename(columns=lambda s: s.lower())
        colnames = sorted(aucs.columns.to_list())
        assert colnames == ["auc", "feature"]
        assert aucs.dtypes["auc"] == "float64"
        assert (aucs.dtypes["feature"] == "object") or \
               (aucs.dtypes["feature"] == "string")
        aucs.to_json("aucs.json")

    validate_aucs(auc_measurements)


if __name__ == "__main__":
    function1("NumberRecognitionBigger.mat")
    function2("cleaned_dataset.csv")

