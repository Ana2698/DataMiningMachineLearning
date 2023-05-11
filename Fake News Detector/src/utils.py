import nltk
import numpy as np
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          GlobalMaxPool1D)
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def data_preparation(data_path1, data_path2, data_path3):
    # This functions performs the data cleaning it takes the dataset filepath
    # as input and returns cleaned data
    true_news = pd.read_csv(data_path1)
    fake_news = pd.read_csv(data_path2)
    validation_test_set2 = pd.read_csv(data_path3, encoding='unicode_escape')
    true_news['class'] = 1
    fake_news['class'] = 0
    true_news_test = true_news.sample(frac=0.1, random_state=0)
    fake_news_test = fake_news.sample(frac=0.1, random_state=0)
    true_news.drop(true_news_test.index, inplace=True)
    fake_news.drop(fake_news_test.index, inplace=True)
    news = pd.concat([true_news, fake_news])
    validation_test_set1 = pd.concat([true_news_test, fake_news_test])
    return news, validation_test_set1, validation_test_set2


"""
common text preprocessing techniques reference
Url: https://towardsdatascience.com/text-preprocessing-in-
natural-language-processing-using-python-6113ff5decd8
Author: Harshith
"""


def stopword_remover_stemmer(text):
    # This is a helper function for the data preprocessing function
    word_list = []
    # print(type(text))
    # text_list = word_tokenize(text)
    stemmer = SnowballStemmer(language='english')
    stop_word_list = list(stopwords.words('english'))
    for word in text.split():
        if word not in stop_word_list:
            word_list.append(stemmer.stem(word))
    # print(word_list)
    output = ' '.join(word_list)
    return output


def data_preprocessing(data, option):
    # This performs data preprocessing it takes the cleaned data as input
    # The option argument differentiates the preprocessing for the ISOT data and the custom created data
    # It returns the preprocessed data.
    if option == 1:  # Preprocessing ISOT data
        data_cleaned = pd.DataFrame()
        data['text'] = data['title'] + ' ' + data['text']
        # reference for the regex expression for eliminating punctuation from text
        # Url: (https://towardsdatascience.com/remove-punctuation-pandas-3e461efe9584)
        # Author: Giorgos Myrianthous Date: 16/04/2023
        # *******************************************************************
        data_cleaned['text'] = data['text'].str.replace('(Reuters)', '').str.replace(r'[^\w\s]+', '').str.lower().apply(
            stopword_remover_stemmer)
        # *******************************************************************
        data_cleaned['class'] = data['class']
    else:  # Preprocessing custom created data
        data_cleaned = pd.DataFrame()
        data_cleaned['text'] = data['text'].str.replace(r'[^\w\s]+', '').str.lower().apply(stopword_remover_stemmer)
        data_cleaned['class'] = data['class']
    return data_cleaned


def test_data_spilt(val_data_cleaned):
    # This splits the custom data into X and y
    X_test_final = val_data_cleaned['text']
    y_test_final = val_data_cleaned['class']
    return X_test_final, y_test_final


def data_split(data_cleaned):
    # This splits the ISOT data into X and y and then performs train_test_split
    X = data_cleaned['text']
    y = data_cleaned['class']
    X_train_unvectorized, X_test_unvectorized, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    return X_train_unvectorized, X_test_unvectorized, y_train, y_test


"""
reference of feature extraction techniques
Title: A Complete Guide on Feature Extraction Techniques
Url:https://www.analyticsvidhya.com/blog/2022/05/
a-complete-guide-on-feature-extraction-techniques/
Author: Shankar
"""


def data_vectorizer_tfidf(X_train_unvectorized, X_test_unvectorized):
    # This performs feature extraction on split data and returns vectorized data
    # along with the fitted vectorizer object for performing testing

    tfidvectorizer = TfidfVectorizer(max_df=0.6, max_features=10000)
    X_train_vectorized = tfidvectorizer.fit_transform(X_train_unvectorized)
    X_test_vectorized = tfidvectorizer.transform(X_test_unvectorized)
    X_train = X_train_vectorized.toarray()
    X_test = X_test_vectorized.toarray()
    return X_train, X_test, tfidvectorizer


def data_vectorizer_count(X_train_unvectorized, X_test_unvectorized):
    # This performs feature extraction on split data and returns vectorized data
    # along with the fitted vectorizer object for performing testing

    countvectoriser = CountVectorizer(max_df=0.8, max_features=10000)
    X_train_vectorized = countvectoriser.fit_transform(X_train_unvectorized)
    X_test_vectorized = countvectoriser.transform(X_test_unvectorized)
    X_train = X_train_vectorized.toarray()
    X_test = X_test_vectorized.toarray()
    return X_train, X_test, countvectoriser


# reference for the regex expression for eliminating punctuation from text
# Url: (https://towardsdatascience.com/sentiment-analysis-using-lstm-and-glove-embeddings-99223a87fe8e)
# Author: Ketan Vaidya Date: 20/11/2022
# *******************************************************************

def data_tokenizer(X_train_unvectorized, X_test_unvectorized):
    # This performs tokenization on split data and returns tokenized and padded data
    # along with the fitted tokenizer object for performing testing and also data embedding.
    sent_len = 20
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train_unvectorized)
    X_train_tokenized = tokenizer.texts_to_sequences(X_train_unvectorized)
    X_train_padded = pad_sequences(X_train_tokenized, maxlen=sent_len)
    X_test_tokenized = tokenizer.texts_to_sequences(X_test_unvectorized)
    X_test_padded = pad_sequences(X_test_tokenized, maxlen=sent_len)
    return X_train_padded, X_test_padded, tokenizer


def data_embedding(tokenizer):
    # This creates the embedded matrix with the pre-trained word embedding

    sent_len = 20
    embedding_dict = {}
    words_index = tokenizer.word_index
    # Link to download th GloVe embeddings
    # https://nlp.stanford.edu/projects/glove/
    with open('/content/drive/MyDrive/project/src/glove.6B.300d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vectors
    f.close()
    vocab_size = len(words_index) + 1
    embed_vector_len = embedding_dict['moon'].shape[0]
    embedding_matrix = np.zeros((vocab_size, embed_vector_len))
    for word, i in words_index.items():
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i, :] = emb_vec
    return vocab_size, sent_len, embedding_matrix, embed_vector_len


# *********************************************************************


def classifier_models():
    # defines the 3 classifer models
    models = {'Linear-SVC': LinearSVC(random_state=0),
              'Linear Regression': LogisticRegression(random_state=0),
              'MultinomialNB': MultinomialNB()}
    return models


def model_scores(X_train, X_test, y_train, y_test, option):
    # This fits the classifier models defined and performs prediction on test data
    # and gives train and test accuracies and also creates classification reports

    model_test_acc = []
    model_train_acc = []
    models = classifier_models()
    if option == 1:
        file = open('/content/drive/MyDrive/project/outputs/test_classifier_report_with_tfidf.txt', 'a')
        file.write('CLASSIFIER with TF-IDF vectorizer TEST REPORT        \n')
    else:
        file = open('/content/drive/MyDrive/project/outputs/test_classifier_report_with_count.txt', 'a')
        file.write('CLASSIFIER with Count Vectorizer TEST REPORT        \n')
    for key in models:
        model = models[key].fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        models[key] = model
        train_acc = accuracy_score(y_train, train_prediction)
        test_acc = accuracy_score(y_test, test_prediction)
        train_acc = train_acc * 100
        test_acc = test_acc * 100
        model_train_acc.append(train_acc)
        model_test_acc.append(test_acc)
        classification = classification_report(y_test, test_prediction)
        file.write(key)
        file.write('     \n')
        file.write("Train Accuracy       \n")
        file.write(str(train_acc))
        file.write('     \n')
        file.write(classification)
    file.close()
    return model_train_acc, model_test_acc, models


"""
some of the references used for the LSTM models
Url: (https://www.kaggle.com/code/samarthsarin/lstm-and-convolution1d-ensemble-with-glove)
Author: Samarth Sarin
Url: (https://analyticsindiamag.com/complete-guide-to-bidirectional-lstm-with-python-codes/)
Author: Yugesh Verma
"""


def basic_lstm(X_train_padded, X_test_padded, y_train, y_test):
    training_accuracy = []
    testing_accuracy = []
    model = Sequential()
    model.add(Embedding(10000, 1000, input_length=20))
    model.add(LSTM(units=500, return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train_padded, y_train, batch_size=100, validation_data=(X_test_padded, y_test), epochs=20)
    # print("Accuracy of the model on Testing Data is:", model.evaluate(X_test, y_test)[1]*100, "%")
    temp, train_acc = model.evaluate(X_train_padded, y_train)
    temp, test_acc = model.evaluate(X_test_padded, y_test)
    train_acc = train_acc * 100
    test_acc = test_acc * 100
    training_accuracy.append(train_acc)
    testing_accuracy.append(test_acc)
    return model, training_accuracy, testing_accuracy


def lstm_em(X_train_padded, X_test_padded, y_train, y_test, vocab_size, sent_len, embedding_matrix, embed_vector_len):
    training_accuracy = []
    testing_accuracy = []
    units_lstm1 = 200
    epochs = 20
    model1 = Sequential()
    model1.add(
        Embedding(input_dim=vocab_size, output_dim=embed_vector_len, input_length=sent_len, weights=[embedding_matrix],
                  trainable=False))
    model1.add(LSTM(units=units_lstm1, return_sequences=True))
    model1.add(Dropout(0.1))
    model1.add(Dense(64, activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(GlobalMaxPool1D())
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.summary()
    model1.fit(X_train_padded, y_train, batch_size=100, validation_data=(X_test_padded, y_test), epochs=epochs)
    # print("Accuracy of the model on Testing Data is:", model.evaluate(X_test_padded,y_test)[1]*100, "%")
    temp, train_acc = model1.evaluate(X_train_padded, y_train)
    temp, test_acc = model1.evaluate(X_test_padded, y_test)
    train_acc = train_acc * 100
    test_acc = test_acc * 100
    training_accuracy.append(train_acc)
    testing_accuracy.append(test_acc)
    return model1, training_accuracy, testing_accuracy


def bi_lstm(X_train_padded, X_test_padded, y_train, y_test, vocab_size, sent_len, embedding_matrix, embed_vector_len):
    training_accuracy = []
    testing_accuracy = []
    units_lstm1 = 200
    epochs = 20
    model3 = Sequential()
    model3.add(
        Embedding(input_dim=vocab_size, output_dim=embed_vector_len, input_length=sent_len, weights=[embedding_matrix],
                  trainable=False))
    model3.add(Dropout(0.1))
    model3.add(Bidirectional(LSTM(units=units_lstm1, return_sequences=True)))
    model3.add(Dropout(0.1))
    model3.add(Dense(50, activation='relu'))
    model3.add(Dropout(0.1))
    model3.add(GlobalMaxPool1D())
    model3.add(Dense(1, activation='sigmoid'))
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model3.summary()
    model3.fit(X_train_padded, y_train, batch_size=100, validation_data=(X_test_padded, y_test), epochs=epochs)
    temp, train_acc = model3.evaluate(X_train_padded, y_train)
    temp, test_acc = model3.evaluate(X_test_padded, y_test)
    train_acc = train_acc * 100
    test_acc = test_acc * 100
    training_accuracy.append(train_acc)
    testing_accuracy.append(test_acc)
    return model3, training_accuracy, testing_accuracy
