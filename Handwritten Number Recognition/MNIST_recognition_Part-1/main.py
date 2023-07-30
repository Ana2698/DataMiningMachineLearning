from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# The code for question1 is adapted from
# Derek's instruction videos and instruction manuals
def function1(path):
    test_errors = []
    prediction_accuracy = []
    # data is loaded
    num_data = loadmat(path)
    # train and test datasets are stored in different variables
    train8 = num_data['imageArrayTraining8']
    test8 = num_data['imageArrayTesting8']
    train9 = num_data['imageArrayTraining9']
    test9 = num_data['imageArrayTesting9']
    # train and test dataset for 8s and 9s are combined and put
    # into xtrain and xtest
    xtrain = np.concatenate([train8, train9], axis=2)
    xtest = np.concatenate([test8, test9], axis=2)
    # shape of xtrain is changed and sample and features are separeted to form the X_train set
    xtrain = xtrain.transpose(2, 0, 1)
    train_sample = xtrain.shape[0]
    train_features = np.prod(xtrain.shape[1:])
    X_train = xtrain.reshape(train_sample, train_features)
    # shape of xtest is changed and sample and features are separeted to form the X_test set
    xtest = xtest.transpose(2, 0, 1)
    test_sample = xtest.shape[0]
    test_features = np.prod(xtest.shape[1:])
    X_test = xtest.reshape(test_sample, test_features)
    # training and testing labels are created
    label_1_train = np.ones(train8.shape[-1])
    label_0_train = np.zeros(train9.shape[-1])
    y_train = np.concatenate([label_0_train, label_1_train])
    label_1_test = np.ones(test8.shape[-1])
    label_0_test = np.zeros(test9.shape[-1])
    y_test = np.concatenate([label_0_test, label_1_test])
    # list of k values
    kvalues = [i for i in range(1, 21)]
    # kNN classifier loop for k values 1-20
    for k in kvalues:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Accuracy corresponding to each k-value is recorded
        prediction_accuracy.append(accuracy_score(y_test, y_pred))
    for i in range(len(prediction_accuracy)):
        # Test error corresponding to  each k-value is calculated and recorded
        test_errors.append((1 - prediction_accuracy[i]) * 100)
    # plot of k-value vs test error is generated
    plt.figure(figsize=(7, 5))
    plt.plot(kvalues, test_errors, linestyle='-.', color='r', marker='h', markerfacecolor='r')
    plt.xticks(kvalues)
    plt.title('K Values VS Test Error Rates')
    plt.xlabel('K Value')
    plt.ylabel('Test Error Rates')
    plt.savefig('knn_q1.png')
    plt.show()

    def save(errors) -> None:
        arr = np.array(errors)
        if len(arr.shape) > 2 or (len(arr.shape) == 2 and 1 not in arr.shape):
            raise ValueError("Invalid output shape. Output should be an array ""that can be unambiguously "
                             "raveled/squeezed.")
        if arr.dtype not in [np.float64, np.float32, np.float16]:
            raise ValueError("Your error rates must be stored as float values.")
        arr = arr.ravel()
        if len(arr) != 20 or (arr[0] >= arr[-1]):
            raise ValueError("There should be 20 error values, with the first value ""corresponding to k=1, and the "
                             "last to k=20.")
        if arr[-1] >= 2.0:
            raise ValueError("Final array value too large. You have done something ""very wrong (probably relating to "
                             "standardizing).")
        if arr[-1] < 0.8:
            raise ValueError("You probably have not converted your error rates to percent values.")
        outfile = Path(__file__).resolve().parent / "errors.npy"
        np.save(outfile, arr, allow_pickle=False)
        print("Error rates succesfully saved to {outfile }")

    save(test_errors)


if __name__ == "__main__":
    function1("NumberRecognitionDataset.mat")
