from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pickle
import shutil
import os
import numpy as np 
from utils import config
from utils.data_preparation import prepareData


def train ():
    try:
        if os.path.isfile(config.EMBEDDING_PATH):
            os.remove(config.EMBEDDING_PATH)
            prepareData(config.EMBEDDING_PATH)
        # load data from disk
        data = pickle.loads(open(config.EMBEDDING_PATH, 'rb').read())
        #grab data and labels
        X_train = data['data']
        y_train = data['names']
        # encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        # perform training
        grid = {'C': [.001, .01, 1.0, 10.0, 100]}
        recognizer = GridSearchCV(SVC(kernel="linear", probability=True), grid, cv=10)
        recognizer.fit(X_train, y_train)
        # save model and label encoder to disk
        with open(config.MODEL_PATH, 'wb') as f:
            f.write(pickle.dumps(recognizer.best_estimator_))

        with open(config.LABEL_PATH, 'wb') as f:
            f.write(pickle.dumps(le))
    except Exception as e:
        raise e


if __name__ == '__main__':
    train()