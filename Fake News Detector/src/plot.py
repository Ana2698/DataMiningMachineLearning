import matplotlib.pyplot as plt
import numpy as np


def classifier_plot(model_train_acc, model_test_acc, option):
    classifier_names = ['Linear SVC', 'Linear Regression', 'MultinomialNB']
    # reference for plot code
    # Url: https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-
    # matplotlib-in-python/
    # Author: Greeshmanalla
    # *********************************************************************
    X_axis = np.arange(len(classifier_names))
    fig = plt.figure()
    plt.bar(X_axis - 0.2, model_train_acc, 0.4, label="train")
    plt.bar(X_axis + 0.2, model_test_acc, 0.4, label="test")
    plt.ylim(60, 100)
    plt.xticks(X_axis, classifier_names)
    # **********************************************************************
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy(in %)')
    plt.legend()
    if option == 1:
        fig.suptitle("Classifier models Training and Testing Accuracies using TF-IDF Vectorizer")
        fig.savefig('/content/drive/MyDrive/project/figs/classifier_train_test_accuracy_tfidf.png')
    else:
        fig.suptitle("Classifier models Training and Testing Accuracies using CountVectorizer")
        fig.savefig('/content/drive/MyDrive/project/figs/classifier_train_test_accuracy_count.png')


def neuralnet_plot(training_accuracy_lstm_b, testing_accuracy_lstm_b, training_accuracy_em, testing_accuracy_em,
                   training_accuracy_lstm_bi, testing_accuracy_lstm_bi):
    training_accuracy_lstm_b.append(training_accuracy_em)
    training_accuracy_lstm_b.append(training_accuracy_lstm_bi)
    testing_accuracy_lstm_b.append(testing_accuracy_em)
    testing_accuracy_lstm_b.append(testing_accuracy_lstm_bi)
    nn_names = ['Basic LSTM', 'LSTM with Embedding', 'Bidirectional LSTM']
    # reference for plot code
    # Url: https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-
    # matplotlib-in-python/
    # Author: Greeshmanalla
    # *********************************************************************
    X_axis = np.arange(len(nn_names))
    fig = plt.figure()
    plt.bar(X_axis - 0.2, training_accuracy_lstm_b, 0.4, label="train")
    plt.bar(X_axis + 0.2, testing_accuracy_lstm_b, 0.4, label="test")
    # plt.bar(X_axis + 0.2, testing_accuracy_lstm_b, 0.6,label="test")
    plt.ylim(40, 100)
    plt.xticks(X_axis, nn_names)
    # **********************************************************************
    plt.xlabel('Neural Network LSTM models')
    plt.ylabel('Accuracy(in %)')
    plt.legend()
    fig.suptitle("LSTM Neural Network models Train and Test Accuracies ")
    fig.savefig('/content/drive/MyDrive/project/figs/LSTM_train_test_accuracy.png')


def classifier_validation_plot(model_validationACC_1, option):
    classifier_names = ['Linear SVC', 'Linear Regression', 'MultinomialNB']
    fig = plt.figure()
    plt.bar(classifier_names, model_validationACC_1)
    plt.ylim(80, 100)
    plt.xticks(classifier_names)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy(in %)')
    if option == 1:
        fig.suptitle("Classifier model Validation Accuracies using TF-IDF Vectorizer with test set 1")
        fig.savefig('/content/drive/MyDrive/project/figs/classifier_validation_accuracy_tfidf1.png')
    else:
        fig.suptitle("Classifier model Validation Accuracies using CountVectorizer with test set 1")
        fig.savefig('/content/drive/MyDrive/project/figs/classifier_validation_accuracy_count1.png')


def classifier_validation_plot_2(model_validationACC_2, option):
    classifier_names = ['Linear SVC', 'Linear Regression', 'MultinomialNB']
    fig = plt.figure()
    plt.bar(classifier_names, model_validationACC_2)
    plt.ylim(20, 80)
    plt.xticks(classifier_names)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy(in %)')
    if option == 1:
        fig.suptitle("Classifier model Validation Accuracies using TF-IDF Vectorizer with test set 2")
        fig.savefig('/content/drive/MyDrive/project/figs/classifier_validation_accuracy_tfidf2.png')
    else:
        fig.suptitle("Classifier model Validation Accuracies using CountVectorizer with test set 2")
        fig.savefig('/content/drive/MyDrive/project/figs/classifier_validation_accuracy_count2.png')


def neuralnet_validation_plot(basic_val_acc, lstm_em_val_acc, bi_lstm_val_acc):
    val_acc = [basic_val_acc, lstm_em_val_acc, bi_lstm_val_acc]
    nn_names = ['Basic LSTM', 'LSTM with Embedding', 'Bidirectional LSTM']
    fig = plt.figure()
    plt.bar(nn_names, val_acc)
    plt.ylim(70, 100)
    plt.xticks(nn_names)
    plt.xlabel('Neural Network LSTM models')
    plt.ylabel('Accuracy(in %)')
    fig.suptitle("Neural Network models Validation Accuracies with test set 1")
    fig.savefig('/content/drive/MyDrive/project/figs/LSTM_validation_accuracy.png')


def neuralnet_validation_plot_2(basic_val_acc_2, lstm_em_val_acc_2, bi_lstm_val_acc_2):
    val_acc = [basic_val_acc_2, lstm_em_val_acc_2, bi_lstm_val_acc_2]
    nn_names = ['Basic LSTM', 'LSTM with Embedding', 'Bidirectional LSTM']
    fig = plt.figure()
    plt.bar(nn_names, val_acc)
    plt.ylim(20, 70)
    plt.xticks(nn_names)
    plt.xlabel('Neural Network LSTM models')
    plt.ylabel('Accuracy(in %)')
    fig.suptitle("Neural Network models Validation Accuracies with test set 2")
    fig.savefig('/content/drive/MyDrive/project/figs/LSTM_validation_accuracy2.png')
