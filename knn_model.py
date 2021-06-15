from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utility import get_df, TRAIN_PATH, TEST_PATH


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
test_count = count_vect.transform(test_df.text)
print(count_vect.get_feature_names())

tfidf_transformer = TfidfTransformer()

train_tfidf = tfidf_transformer.fit_transform(train_counts)
test_tfidf = tfidf_transformer.transform(test_count)

print('k=15')
model = KNeighborsClassifier(n_neighbors=15)
model.fit(train_tfidf, train_df.category)
predicted = model.predict(test_tfidf)
print(confusion_matrix(test_df.category, predicted))
print(model.score(test_tfidf, test_df.category))

print('K=5')
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_tfidf, train_df.category)
predicted = model.predict(test_tfidf)
print(confusion_matrix(test_df.category, predicted))
print(model.score(test_tfidf, test_df.category))


print('K=1')
model = KNeighborsClassifier(n_neighbors=1)
model.fit(train_tfidf, train_df.category)
predicted = model.predict(test_tfidf)
print(confusion_matrix(test_df.category, predicted))
print(model.score(test_tfidf, test_df.category))
