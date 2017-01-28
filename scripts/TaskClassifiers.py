from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle


# A Class which handles saving and loading a set of classifiers
class TextClassifiers():
    def __init__(self, KNN, SVM, GNB, MNB, DT, RF):
        self.KNN = KNN
        self.SVM = SVM
        self.GNB = GNB
        self.MNB = MNB
        self.DT = DT
        self.RF = RF

    # Returns the result of predicting the class the sample feature vector
    # against the specified model
    def classify(self, model_name, sample):
        model_name = model_name.lower()
        if model_name == 'knn':
            return self.KNN.predict(sample)
        elif model_name == 'svm':
            return self.SVM.predict(sample)
        elif model_name == 'gnb':
            return self.GNB.predict(sample)
        elif model_name == 'mnb':
            return self.MNB.predict(sample)
        elif model_name == 'dt':
            return self.DT.predict(sample)
        elif model_name == 'rf':
            return self.RF.predict(sample)
        else:
            raise NotImplementedError('not an implemented classifer')

    # Saves the models to disk
    def save(self, filename):
        with open(filename, 'wb') as modelfile:
            pickle.dump(self, modelfile)

    # Loads the models from disk
    @staticmethod
    def load(filename):
        tc = None
        with open(filename, 'rb') as modelfile:
            tc = pickle.load(modelfile)
        return tc