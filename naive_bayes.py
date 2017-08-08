import pandas as pd
from time import time
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB
from nltk.stem import SnowballStemmer


data = pd.read_csv("spam.csv",encoding='latin-1')
data = data.rename(columns={"v1": "label", "v2": "text"})


"Analysing data"
print data.describe()


"Deleting unnecessary columns"
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)


"Splitting data"
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(data["text"],
                                                                                             data["label"],
                                                                                             test_size=0.2,
                                                                                             random_state=42)
#print features_train
def pre_process(data):

    stemmed_list = []
    for text in data:
        exclude = set(string.punctuation)
        text = ''.join(ch for ch in text if ch not in exclude)
        stemmer = SnowballStemmer("english")
        stemmed = [stemmer.stem(w) for w in text.split()]
        words = " ".join(stemmed)
        stemmed_list = stemmed_list + [words]
    return stemmed_list


features_train = pre_process(features_train)
features_test = pre_process(features_test)


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed = vectorizer.transform(features_test)

#print features_train_transformed
#print features_train_transformed.shape
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
'''print '--------------------------------------------------------------------'
print features_train_transformed
print features_train_transformed.shape
print labels_train.shape'''
features_test_transformed = selector.transform(features_test_transformed).toarray()
'''print features_test_transformed.shape
print labels_test.shape
print type(features_train_transformed)
print type(labels_train)'''


clf = GaussianNB()
t0 = time()
clf.fit(features_train_transformed, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test_transformed)
print "predicting time:", round(time()-t1, 3), "s"
print clf.score(features_test_transformed, labels_test)
