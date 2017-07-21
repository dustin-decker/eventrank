import re
import numpy as np
import copy
from numpy import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import NuSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing


corpus = []
corpus_labels = []

for i in range(1,25):
    corpus.append("some-event{0} lol-srcevil{0} lamb".format(i))
    corpus_labels.append('true')
    corpus.append("some-event{0} lol-srcgood{0} lamb".format(i))
    corpus_labels.append('false')

for i in range(1,8):
    corpus.append("some-event{0} lol-srcevil{0} lamb".format(i))
    corpus_labels.append('false')



host = re.compile('([a-z]{2,10})-([a-z]{2,18})([0-9]{1,3})')


def tokenize(event):
    tokens = []
    for item in event.split(" "):
        if len(item) > 2:
            if host.match(item):
                res = host.search(item)
                # add the full hostname, the datacenter, and the serverclass, but not the node number
                tokens.extend([res.group(0), res.group(1), res.group(2)])
            else:
                tokens.append(item)
    return tokens

X_train = np.array(corpus)
y_train_text = np.array(corpus_labels)

X_test = np.array(['some-event2 bcd-srcgood2 lamb','some-event2 bcd-srcevil3 lamb','some-event2 bcd-srcevil33 lamb'])

mlb = preprocessing.LabelBinarizer()
Y = mlb.fit_transform(y_train_text)

classifier = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', NuSVC(probability=True))])

classifier.fit(X_train, Y.ravel())
predicted = classifier.predict_proba(X_test)
all_labels = mlb.inverse_transform(predicted)
#all_labels = lb.inverse_transform(predicted)

for item, labels, prediction in zip(X_test, all_labels, predicted):
    print('%s => %s => %s' % (item, labels, prediction))