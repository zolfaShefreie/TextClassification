import os
import pandas as pd
import parsivar
import codecs
# from sets import Set
from hazm import Normalizer
from sklearn import preprocessing
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_PATH = './Dataset/Train'
TEST_PATH = './Dataset/Test'


def get_df(path):
    list_row = list()
    list_of_file_dir = os.listdir(path)
    for each in list_of_file_dir:
        full_path = path + '/' + each
        if os.path.isdir(full_path):
            list_of_entry = os.listdir(full_path)
            for entry in list_of_entry:
                file_path = full_path + '/' + entry
                if not os.path.isdir(file_path):
                    file = open(file_path, encoding='utf8')
                    text = preprocess_text(file.read())
                    # print('hehe')
                    list_row.append({'category': each, 'text': text})
                    file.close()
    return pd.DataFrame(list_row)


def preprocess_text(text):
    text = parsivar.Normalizer().normalize(text)
    tokenizer = parsivar.Tokenizer()
    stop_words = get_stop_words()
    chars = get_characters()
    chars = list(chars)
    chars += [';', ',', '&', '?', '.', '%']
    for each in chars:
        text = text.replace(each, ' ')
    tokens = tokenizer.tokenize_words(text)
    stemmer = parsivar.FindStems()
    new_text = str()
    pass_word = list()
    ignor_word = list()
    for each in tokens:
        new_word = stemmer.convert_to_stem(each)
        if '&' in new_word:
            new_word = new_word.split('&')[0]
        if new_word in stop_words:
            ignor_word.append(new_word)
            continue
        if new_word in [';', ',', '&', '?', 'amp', 'nbsp', '.', 'o']:
            ignor_word.append(new_word)
            continue
        if re.match(r'^\d+$', new_word):
            ignor_word.append(new_word)
            continue
        pass_word.append(new_word)
        new_text += ' ' + new_word
    # print(set(pass_word))
    # print("********************************************************************")
    # print(set(ignor_word))
    # print("__________________________________________________________________________")
    # print(new_text)
    new_text = new_text.replace('‌', '')
    return new_text


def get_characters():
    f = open('./chars', encoding='utf-8')
    chars = f.read()
    return set([w for w in chars.split('\n') if w])


def get_stop_words():
    nmz = parsivar.Normalizer()
    f = open('./persian', encoding='utf-8')
    words = f.read()
    # words = codecs.open('./persian', encoding='utf-8').read()
    return set([nmz.normalize(w) for w in words.split('\n') if w])


if __name__ == "__main__":
    train_df = get_df(TRAIN_PATH)
    train_x_df = train_df[['text']]
    train_y_df = train_df['category']
    le = preprocessing.LabelEncoder()
    le.fit(train_df['category'].unique())
    train_df['category'] = le.transform(train_df['category'])
    test_df = get_df(TEST_PATH)
    test_df['category'] = le.transform(test_df['category'])
    merged_df = train_df.append(test_df)
    print(merged_df.shape, train_df.shape)

    count_vect = CountVectorizer(max_features=500)
    count_vect.fit(merged_df.text)
    X_train_counts = count_vect.transform(train_df.text)
    print(count_vect.get_feature_names())
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    model = MultinomialNB(alpha=1)
    model.fit(X_train_counts, train_df.category)
    test_count = count_vect.transform(test_df.text)
    test_tfitf = tfidf_transformer.transform(test_count)
    predicted = model.predict(test_count)
    # print(model.predict_proba(test_count))

    print(confusion_matrix(test_df.category, predicted))
    print(classification_report(test_df.category, predicted))

    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(X_train_tfidf, train_df.category)
    predicted = model.predict(test_tfitf)
    # print(np.mean(predicted == test_df.category))

    print(confusion_matrix(test_df.category, predicted))
    print(model.score(test_tfitf, test_df.category))
    # print(classification_report(test_df.category, predicted))

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_tfidf, train_df.category)
    predicted = model.predict(test_tfitf)
    # print(np.mean(predicted == test_df.category))

    print(confusion_matrix(test_df.category, predicted))
    print(model.score(test_tfitf, test_df.category))
    # print(classification_report(test_df.category, predicted))

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train_tfidf, train_df.category)
    predicted = model.predict(test_tfitf)

    # print(np.mean(predicted == test_df.category))
    print(confusion_matrix(test_df.category, predicted))
    print(model.score(test_tfitf, test_df.category))
    # print(classification_report(test_df.category, predicted))


    # text_clf = Pipeline([('vect', CountVectorizer(max_features=500)),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf', MultinomialNB(alpha=1))])
    # text_clf = text_clf.fit(train_x_df.text, train_y_df)
    # # clf = MultinomialNB().fit(X_train_tfidf, train_y_df)
    # test_df = get_df(TEST_PATH)
    # predicted = text_clf.predict(test_df.text)
    # test_df['category'] = le.transform(test_df['category'])
    # print(np.mean(predicted == test_df.category))
    # text_clf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', KNeighborsClassifier(n_neighbors=15)),
    # ])
    # text_clf = text_clf.fit(train_x_df.text, train_y_df)
    # predicted = text_clf.predict(test_df.text)
    # print(np.mean(predicted == test_df.category))


