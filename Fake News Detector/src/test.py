from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score


def classifier_validation_test(X_test_val, y_test_val, vectorizer, classifier_models):
    model_validationACC = []
    X_test_final_vectorized = vectorizer.transform(X_test_val)
    for key, value in classifier_models.items():
        prediction = value.predict(X_test_final_vectorized)
        acc = accuracy_score(y_test_val, prediction)
        acc = acc * 100
        model_validationACC.append(acc)
    return model_validationACC


def basic_lstm_validation_test(X_test_val, y_test_val, tokenizer, model):
    X_test_final_tokenized = tokenizer.texts_to_sequences(X_test_val)
    X_test_val = pad_sequences(X_test_final_tokenized, maxlen=20)
    temp, basic_val_acc = model.evaluate(X_test_val, y_test_val)
    basic_val_acc = basic_val_acc * 100
    return basic_val_acc


def lstm_em_validation_test(X_test_val, y_test_val, tokenizer, model1):
    X_test_final_tokenized = tokenizer.texts_to_sequences(X_test_val)
    X_test_val = pad_sequences(X_test_final_tokenized, maxlen=20)
    temp, lstm_em_val_acc = model1.evaluate(X_test_val, y_test_val)
    lstm_em_val_acc = lstm_em_val_acc * 100
    return lstm_em_val_acc


def bi_lstm_validation_test(X_test_val, y_test_val, tokenizer, model3):
    X_test_final_tokenized = tokenizer.texts_to_sequences(X_test_val)
    X_test_val = pad_sequences(X_test_final_tokenized, maxlen=20)
    temp, bi_lstm_val_acc = model3.evaluate(X_test_val, y_test_val)
    bi_lstm_val_acc = bi_lstm_val_acc * 100
    return bi_lstm_val_acc
