from src.utils import data_preparation
from src.utils import data_preprocessing
from src.utils import test_data_spilt
from src.utils import data_split
from src.utils import data_vectorizer_tfidf
from src.utils import data_vectorizer_count
from src.utils import model_scores
from src.utils import data_tokenizer
from src.utils import basic_lstm
from src.utils import data_embedding
from src.utils import lstm_em
from src.utils import bi_lstm
from src.test import basic_lstm_validation_test
from src.test import lstm_em_validation_test
from src.test import classifier_validation_test
from src.plot import classifier_plot
from src.plot import neuralnet_plot
from src.plot import classifier_validation_plot
from src.plot import classifier_validation_plot_2
from src.plot import neuralnet_validation_plot
from src.plot import neuralnet_validation_plot_2
from src.test import bi_lstm_validation_test
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    data_path1 = "/content/drive/MyDrive/project/data/True.csv"  # Part of ISOT Fake news dataset
    data_path2 = "/content/drive/MyDrive/project/data/Fake.csv"  # Part of ISOT Fake news dataset
    data_path3 = "/content/drive/MyDrive/project/data/validation_test_set2.csv"  # Custom news dataset
    news, validation_test_set1, validation_test_set2 = data_preparation(data_path1, data_path2, data_path3)

    data_cleaned = data_preprocessing(news, 1)
    val_data1_cleaned = data_preprocessing(validation_test_set1, 1)
    val_data2_cleaned = data_preprocessing(validation_test_set2, 2)

    # Training Testing and Validation data splitting
    X_train_unvectorized, X_test_unvectorized, y_train, y_test = data_split(data_cleaned)
    X_test_val1, y_test_val1 = test_data_spilt(val_data1_cleaned)
    X_test_val2, y_test_val2 = test_data_spilt(val_data2_cleaned)

    # Feature extraction
    X_train_tf, X_test_tf, tfidvectorizer = data_vectorizer_tfidf(X_train_unvectorized, X_test_unvectorized)
    X_train_count, X_test_count, countvectoriser = data_vectorizer_count(X_train_unvectorized, X_test_unvectorized)

    # Data tokenizing and embedding
    X_train_padded, X_test_padded, tokenizer = data_tokenizer(X_train_unvectorized, X_test_unvectorized)
    vocab_size, sent_len, embedding_matrix, embed_vector_len = data_embedding(tokenizer)

    # Building Classifier models and calculating train and test accuracies and test classification reports
    model_train_acc_tf, model_test_acc_tf, classifier_models_tf = model_scores(X_train_tf, X_test_tf, y_train, y_test,
                                                                               1)
    # Plotting train and test accuracy results
    classifier_plot(model_train_acc_tf, model_test_acc_tf, 1)
    # Perfroming Validation test with ISOT Fake news dataset
    model_validationACC_tfidf_1 = classifier_validation_test(X_test_val1, y_test_val1, tfidvectorizer,
                                                             classifier_models_tf)
    # Perfroming Validation test with custom news dataset
    model_validationACC_tfidf_2 = classifier_validation_test(X_test_val2, y_test_val2, tfidvectorizer,
                                                             classifier_models_tf)
    # Plotting Validation test results
    classifier_validation_plot(model_validationACC_tfidf_1, 1)
    classifier_validation_plot_2(model_validationACC_tfidf_2, 1)
    # Building Classifier models and calculating train and test accuracies and test classification reports
    model_train_acc_count, model_test_acc_count, classifier_models_count = model_scores(X_train_count, X_test_count,
                                                                                        y_train, y_test, 0)
    # Plotting train and test accuracy results
    classifier_plot(model_train_acc_tf, model_test_acc_tf, 0)
    # Perfroming Validation test with ISOT Fake news dataset
    model_validationACC_count_1 = classifier_validation_test(X_test_val1, y_test_val1, countvectoriser,
                                                             classifier_models_count)
    # Perfroming Validation test with custom news dataset
    model_validationACC_count_2 = classifier_validation_test(X_test_val2, y_test_val2, countvectoriser,
                                                             classifier_models_count)
    # Plotting Validation test results
    classifier_validation_plot(model_validationACC_count_1, 0)
    classifier_validation_plot_2(model_validationACC_count_2, 0)
    # Building LSTM models calculating testing and training accuracies for all 3 models
    model, training_accuracy_lstm_b, testing_accuracy_lstm_b = basic_lstm(X_train_padded, X_test_padded, y_train,
                                                                          y_test)
    model1, training_accuracy_em, testing_accuracy_em = lstm_em(X_train_padded, X_test_padded, y_train, y_test,
                                                                vocab_size, sent_len, embedding_matrix,
                                                                embed_vector_len)
    model3, training_accuracy_lstm_bi, testing_accuracy_lstm_bi = bi_lstm(X_train_padded, X_test_padded, y_train,
                                                                          y_test, vocab_size, sent_len,
                                                                          embedding_matrix, embed_vector_len)
    # Plotting Train and Test accuracy results of the LSTM models
    neuralnet_plot(training_accuracy_lstm_b, testing_accuracy_lstm_b, training_accuracy_em, testing_accuracy_em,
                   training_accuracy_lstm_bi, testing_accuracy_lstm_bi)
    # Performing Validation tests with ISOT Fake news and custom news dataset
    basic_val_acc_1 = basic_lstm_validation_test(X_test_val1, y_test_val1, tokenizer, model)
    basic_val_acc_2 = basic_lstm_validation_test(X_test_val2, y_test_val2, tokenizer, model)
    lstm_em_val_acc_1 = lstm_em_validation_test(X_test_val1, y_test_val1, tokenizer, model1)
    lstm_em_val_acc_2 = lstm_em_validation_test(X_test_val2, y_test_val2, tokenizer, model1)
    bi_lstm_val_acc_1 = bi_lstm_validation_test(X_test_val1, y_test_val1, tokenizer, model3)
    bi_lstm_val_acc_2 = lstm_em_validation_test(X_test_val2, y_test_val2, tokenizer, model3)
    # Plotting Validation test results on ISOT Fake news and custom news dataset for 3 LSTM models
    neuralnet_validation_plot(basic_val_acc_1, lstm_em_val_acc_1, bi_lstm_val_acc_1)
    neuralnet_validation_plot_2(basic_val_acc_2, lstm_em_val_acc_2, bi_lstm_val_acc_2)
