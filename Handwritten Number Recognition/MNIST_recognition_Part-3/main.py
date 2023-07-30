from typing import Literal

import numpy as np
import pandas as pd
from keras import Input, Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ReLU
from pandas import DataFrame
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras

"""
This function below creates the kfold_cnn,
 kfold_data and kfold_mnist json files.
The code is taken from the instructions
 provide for assignment 3 submission.
"""


def save_kfold(kfold_scores: pd.DataFrame, function: Literal[1, 2]) -> None:
    from pathlib import Path

    from pandas import DataFrame

    COLS = [*[f"fold{i}" for i in range(1, 6)], "mean"]
    INDEX = {
        1: ["cnn", "knn1", "knn5", "knn10"],
        2: ["cnn"],
    }[function]
    outname = {
        1: "kfold_mnist.json",
        2: "kfold_cnn.json",
    }[function]
    outfile = Path(outname)

    # name checks
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError(
            "Argument `kfold_scores` to "
            "`save` must be a pandas DataFrame."
        )
    if kfold_scores.shape[1] != 6:
        raise ValueError("DataFrame must have 6 columns.")
    if df.columns.to_list() != COLS:
        raise ValueError(
            f"Columns are incorrectly named and/or"
            f" incorrectly sorted. Got:\n{df.columns.to_list()}\n"
            f"but expected:\n{COLS}"
        )
    if df.index.name.lower() != "classifier":
        raise ValueError(
            "Index is incorrectly named."
            " Create a proper index using `pd.Series` or "
            "`pd.Index` and use the `name='classifier'` argument."
        )
    idx_items = sorted(df.index.to_list())
    for idx in INDEX:
        if idx not in idx_items:
            raise ValueError(f"You are missing a row with index value {idx}")

    # value range checks
    if function == 1:
        if df.loc["cnn", "mean"] < 0.05:
            raise ValueError(
                "Your CNN error rate is too low. "
                "Make sure you implement the CNN as provided in "
                "the assignment or example code."
            )
        if df.loc[["knn1", "knn5"], "mean"].min() > 0.04:
            raise ValueError(
                "One of your KNN-1 or KNN-5 "
                "error rates is too high. There "
                "is likely an error in your code."
            )
        if df.loc["knn10", "mean"] > 0.047:
            raise ValueError(
                "Your KNN-10 error rate is too high. "
                "There is likely an error in your code."
            )
        df.to_json(outfile)
        print(f"K-Fold error rates for MNIST data "
              f"successfully saved to {outfile}")
        return

    # must be function 2
    df.to_json(outfile)
    print(
        f"K-Fold error rates for custom CNN on "
        f"MNIST data successfully saved to {outfile}"
    )


"""
The function below takes X and y data as input and K-fold splits them
This function is only to perform kfold split on question 1 and 3 data
"""


def k_fold_split(X, y, question):
    cnn_acc = []
    knn_ann_err = []
    knn_err = []
    # cnn_model and knn_ann_model functions are called to get the
    # respective cnn and knn models for question 1.
    model = cnn_model(1)
    knn_models = knn_ann_model(1)
    skf = StratifiedKFold(n_splits=5, random_state=None)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # cnn_score and knn_score functions are called to
        # get the error rates of the models
        cnn_acc.append(cnn_score(X_train, X_test,
                                 y_train, y_test, model))
        knn_err.append(knn_score(X_train, X_test,
                                 y_train, y_test, knn_models))
    cnn_err = [round((1 - i), 3) for i in cnn_acc]
    return knn_err, cnn_err


"""
The function below defines the knn and
 ann models required for questions 1 and 3.
"""


def knn_ann_model(question):
    # The "models" dictionary defines all the classifier being compared.
    models = {
        'knn1': KNeighborsClassifier(n_neighbors=1),
        'knn5': KNeighborsClassifier(n_neighbors=5),
        'knn10': KNeighborsClassifier(n_neighbors=10)
    }
    return models


"""
The function below is used to obtain the
error rates of knn models only for question 1
"""


def knn_score(X_train, X_test, y_train, y_test, models):
    # The "models" dictionary passed as argument
    # it defines all the classifier being compared.
    model_testerror = []
    test_error = []
    # models dictionary is iterated over
    for key, value in models.items():
        value.fit(X_train, y_train.ravel())
        prediction = value.predict(X_test)
        # The test errors for each classifier is
        # calculated and rounded to 3 decimal points,
        # test errors are stored in model_testerror list
        model_testerror.append((1 -
                                accuracy_score(y_test, prediction)
                                ).round(3))
        # The "test_error" list stores test_errors for all 6 classifiers,
        # over all 5 folds
    test_error.append(model_testerror)
    model_testerror = []
    return test_error


"""
The function below is used to obtain the error rates
of CNN model only for question 1
"""


def cnn_score(X_train, X_test, y_train, y_test, model):
    NUM_CLASSES = 10
    BATCH_SIZE = 8
    EPOCHS = 1
    # Reshaping data and converting integer to floats
    X_train = X_train.reshape((X_train.shape[0], 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 28, 28))
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    # Converting to binary classes
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    # model fitting
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS, validation_split=0.1)
    # model evaluation
    temp, acc = model.evaluate(X_test, y_test, verbose=1)
    return acc


# The function below defines the CNN model for question 1 and 4


def cnn_model(function):
    if function == 1:
        NUM_CLASSES = 10
        INPUT_SHAPE = (28, 28, 1)
        OUT_CHANNEL = 1
        model = Sequential([Input(shape=INPUT_SHAPE),
                            Conv2D(OUT_CHANNEL, kernel_size=3,
                                   padding='same', use_bias=True),
                            ReLU(),
                            Flatten(),
                            Dense(NUM_CLASSES, activation='softmax'), ])
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        return model
    if function == 2:
        NUM_CLASSES = 10
        INPUT_SHAPE = (28, 28, 1)
        model = Sequential([Input(shape=INPUT_SHAPE),
                            Conv2D(40, kernel_size=(3, 3),
                                   padding='same', use_bias=True),
                            MaxPooling2D((2, 2)),
                            BatchNormalization(),
                            Dropout(0.2),
                            Conv2D(70, kernel_size=(3, 3),
                                   use_bias=True),
                            BatchNormalization(),
                            Conv2D(100, kernel_size=(3, 3),
                                   use_bias=True),
                            MaxPooling2D((2, 2)),
                            BatchNormalization(),
                            Dropout(0.2),
                            ReLU(),
                            Flatten(),
                            Dense(100, activation='sigmoid'),
                            Dense(NUM_CLASSES, activation='softmax'), ])
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        return model


"""
The function below is used to obtain the error rates
amd predictions of CNN model only for question 4
"""


def cnn_f2_score(X, y):
    NUM_CLASSES = 10
    EPOCHS = 15
    model = cnn_model(4)
    # Reading and reshaping X_test data for prediction.
    mnist_data = loadmat('NumberRecognitionBiggest.mat')
    test_x = mnist_data['X_test']
    x_samples_test, x_features_test = test_x.shape[0], np.prod(
        test_x.shape[1:])
    X_testflat = test_x.reshape(x_samples_test, x_features_test)
    X_testflat = X_testflat.reshape((X_testflat.shape[0], 28, 28))
    X_testflat = X_testflat.astype("float32") / 255
    X_testflat = np.expand_dims(X_testflat, -1)
    # Performing kfold split on the X, y data.
    skf = StratifiedKFold(n_splits=5, random_state=None)
    cnn_f2_list = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Reshaping data and converting integer to floats
        X_train = X_train.reshape((X_train.shape[0], 28, 28))
        X_test = X_test.reshape((X_test.shape[0], 28, 28))
        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        # Converting to binary classes
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        # model fitting
        model.fit(X_train, y_train, batch_size=10,
                  epochs=EPOCHS, validation_split=0.1)
        # model evaluation
        temp, acc = model.evaluate(X_test, y_test, verbose=1)
        cnn_f2_list.append(acc)
    cnn_f2_err = [round((1 - i), 3) for i in cnn_f2_list]
    # model prediction on X_test data provided
    y_pred = model.predict(X_testflat)
    return cnn_f2_err, y_pred


"""
The 3 functions below model_scoredf_q1, model_scoredf_q3, model_scoredf_q4
are used to create the dataframe that contains
the K-Fold error rates for each classifier.
They takes list of test errors return by the respective _score() functions.
And they returns the required dataframe.
"""


def model_scoredf_f1(knn_err, cnn_err):
    classifier_name = ['cnn']
    classifier_names_1 = ['knn1', 'knn5', 'knn10']
    print(cnn_err)
    knn_err_list = [item for sub_list in knn_err for item in sub_list]
    # Dictionary containing the dataframe's column names as key
    # and the dataframe's row values as value.
    df_dict = {'fold1': cnn_err[0],
               'fold2': cnn_err[1],
               'fold3': cnn_err[2],
               'fold4': cnn_err[3],
               'fold5': cnn_err[4]
               }
    df_dict_1 = {'fold1': knn_err_list[0],
                 'fold2': knn_err_list[1],
                 'fold3': knn_err_list[2],
                 'fold4': knn_err_list[3],
                 'fold5': knn_err_list[4]
                 }
    # Dataframe created with the classifier names as the index
    df_0 = pd.DataFrame(df_dict, index=classifier_name)
    df_1 = pd.DataFrame(df_dict_1, index=classifier_names_1)
    frames = [df_0, df_1]
    df = pd.concat(frames)
    df.index.name = 'classifier'
    # Mean of the test errors of all 5 folds for each model is calculated,
    # and stored in a new column in the dataframe created above.
    df['mean'] = df.mean(axis=1, numeric_only=True)
    return df


def model_scoredf_f2(cnn_err):
    classifier_name = ['cnn']
    print(cnn_err)
    # Dictionary containing the dataframe's column names as key
    # and the dataframe's row values as value.
    df_dict = {'fold1': cnn_err[0],
               'fold2': cnn_err[1],
               'fold3': cnn_err[2],
               'fold4': cnn_err[3],
               'fold5': cnn_err[4]
               }
    # Dataframe created with the classifier names as the index
    df = pd.DataFrame(df_dict, index=classifier_name)
    df.index.name = 'classifier'
    # Mean of the test errors of all 5 folds for each model is calculated,
    # and stored in a new column in the dataframe created above.
    df['mean'] = df.mean(axis=1, numeric_only=True).round(3)
    return df


def function1():
    num_data = loadmat('NumberRecognitionBiggest.mat')
    # X, y data read and reshaped
    x_data, y_data = num_data['X_train'], num_data['y_train'].transpose()
    x_samples, x_features = x_data.shape[0], np.prod(x_data.shape[1:])
    X = x_data.reshape(x_samples, x_features)
    # k_fold_split function called to split data
    knn_err, cnn_err = k_fold_split(X, y_data, 1)
    # model_scoredf_q1 function called to
    # create the dataframe for save_kfold
    f1_df = model_scoredf_f1(knn_err, cnn_err)
    save_kfold(f1_df, 1)
    print(f1_df)


def function2():
    from pathlib import Path
    num_data = loadmat('NumberRecognitionBiggest.mat')
    # X, y data read and reshaped
    x_data, y_data = num_data['X_train'], num_data['y_train'].transpose()
    x_samples, x_features = x_data.shape[0], np.prod(x_data.shape[1:])
    X = x_data.reshape(x_samples, x_features)
    # cnn_q4_score function called to split data
    # and perform fit and predict
    cnn_err, y_pred = cnn_f2_score(X, y_data)
    # model_scoredf_f2 function called to
    # create the dataframe for save_kfold
    f2_df = model_scoredf_f2(cnn_err)
    save_kfold(f2_df, 2)
    # Url: (https://towardsdatascience.com/increase-the-accuracy-of-your-
    # cnn-by-following-these-5-tips-i-learned-from-the-kaggle-community-27227ad39554)
    # Author: Patrick Kalkman Date: 01/03/2023
    # *******************************************************************
    y_pred = np.argmax(y_pred, axis=1)
    # *******************************************************************
    print(y_pred.shape)
    np.save(
        Path("predictions.npy"),
        y_pred.astype(np.uint8),
        allow_pickle=False,
        fix_imports=False)


if __name__ == "__main__":
    function1()
    function2()
