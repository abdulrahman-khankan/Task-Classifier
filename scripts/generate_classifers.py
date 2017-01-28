# This files generates different classifiers, trains them on
# the dataset and scores their accuracy against the test set

import dataset_handler
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from TaskClassifiers import TextClassifiers




tag_train, dataset_train = dataset_handler.read_data('../data/dataset_train.csv')
tag_test, dataset_test = dataset_handler.read_data('../data/dataset_test.csv')

# Generate, train and score SVM classifiers
print 'svm classifiers'
clf_svm_lin = svm.SVC(kernel='linear').fit(dataset_train, tag_train)
clf_svm_rbf = svm.SVC().fit(dataset_train, tag_train)
clf_svm_poly = svm.SVC(kernel='poly').fit(dataset_train, tag_train)
clf_svm_sig = svm.SVC(kernel='sigmoid').fit(dataset_train, tag_train)

print '\nlinear svm %s' % clf_svm_lin.score(dataset_test, tag_test)
print 'rbf svm %s' % clf_svm_rbf.score(dataset_test, tag_test)
print 'poly svm %s' % clf_svm_poly.score(dataset_test, tag_test)
print 'sigmoid svm %s' % clf_svm_sig.score(dataset_test, tag_test)

# Generate, train and score KNN classifiers
print '\nknn classifiers'
clf_knn_dist = KNeighborsClassifier(3, 'distance').fit(dataset_train, tag_train)
clf_knn_unif = KNeighborsClassifier(3, 'uniform').fit(dataset_train, tag_train)

print 'knn distance %s' % clf_knn_dist.score(dataset_test, tag_test)
print 'knn uniform %s' % clf_knn_unif.score(dataset_test, tag_test)

# Generate, train and score Naive Bayes classifiers
print '\nnaive bayes'
clf_nb_g = GaussianNB().fit(dataset_train, tag_train)
clf_nb_mn = MultinomialNB().fit(dataset_train, tag_train)

print 'gausian naive bayes %s' % clf_nb_g.score(dataset_test, tag_test)
print 'multinomial naive bayes %s' % clf_nb_mn.score(dataset_test, tag_test)

# Generate, train and score Decision Tree classifier
print  '\ndecision trees'
clf_dt = DecisionTreeClassifier(random_state=0).fit(dataset_train, tag_train)
print 'decision tree %s' % clf_dt.score(dataset_test, tag_test)

# Generate, train and score Random forest classifier
print '\n random forest'
clf_rf = RandomForestClassifier().fit(dataset_train, tag_train)
print 'random forest %s' % clf_rf.score(dataset_test, tag_test)

tc = TextClassifiers(clf_knn_dist, clf_svm_lin, clf_nb_g, clf_nb_mn, clf_dt, clf_rf)
tc.save('../data/TextClassifiers.tc')
