from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from utility import get_df, TRAIN_PATH, TEST_PATH


if __name__ == '__main__':
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
    train_counts = count_vect.transform(train_df.text)
    print(type(train_counts))

    print(count_vect.get_feature_names())

    model = MultinomialNB(alpha=1)
    model.fit(train_counts, train_df.category)
    test_count = count_vect.transform(test_df.text)
    predicted = model.predict(test_count)
    # print(model.score(train_counts, train_df.category))
    print(confusion_matrix(test_df.category, predicted))
    print(classification_report(test_df.category, predicted))
    # print(model.predict_proba(test_count))